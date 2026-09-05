const std = @import("std");

const zml = @import("zml");

const Float32 = zml.floats.Float32;
const Float8E4M3 = zml.floats.Float8E4M3;
const Float4E2M1 = zml.floats.Float4E2M1;

pub const DEFAULT_ALIGN = 16;
pub const DEFAULT_SIMD_SIZE = 16;

fn lsbMask(comptime T: type, comptime bits: usize) T {
    std.debug.assert(bits <= @bitSizeOf(T));
    if (bits == @bitSizeOf(T)) {
        return ~@as(T, 0);
    } else {
        return (@as(T, 1) << bits) - @as(T, 1);
    }
}

test lsbMask {
    try std.testing.expectEqual(lsbMask(u32, 4), 0b1111);
    try std.testing.expectEqual(lsbMask(u32, 0), 0b0);
    try std.testing.expectEqual(lsbMask(u8, 8), 0b11111111);
}

pub const NvFp4Buffers = struct {
    const Self = @This();

    const BLOCK_SIZE = 16;

    nelements: usize,
    fp4block16: [*]align(DEFAULT_ALIGN) const u8,
    block_scale: [*]const u8,
    tensor_scale: f32,

    pub fn validate(fp4block16: []align(DEFAULT_ALIGN) const u8, block_scale: []const u8, tensor_scale: f32) !NvFp4Buffers {
        const nelements = 2 * fp4block16.len;
        if (nelements % BLOCK_SIZE != 0) {
            return error.Invalid;
        }
        const nscales = block_scale.len * @sizeOf(u8);
        if (nscales != nelements / BLOCK_SIZE) {
            return error.Invalid;
        }
        return .{
            .nelements = nelements,
            .fp4block16 = fp4block16.ptr,
            .block_scale = block_scale.ptr,
            .tensor_scale = tensor_scale,
        };
    }

    fn nBlocks(self: *const Self) usize {
        return self.nelements / BLOCK_SIZE;
    }
};

pub fn dequantF32Scalar(input: NvFp4Buffers, output: []align(DEFAULT_ALIGN) f32) void {
    std.debug.assert(output.len == input.nelements);
    const bytes_per_block = NvFp4Buffers.BLOCK_SIZE / 2;
    const fp4_size: u8 = 4;
    const fp4_mask = comptime lsbMask(u8, fp4_size);
    const tensor_scale = input.tensor_scale;

    for (0..input.nBlocks()) |blk| {
        const block_scale_packed: Float8E4M3 = @bitCast(input.block_scale[blk]);
        const block_scale = block_scale_packed.toF32();

        for (0..bytes_per_block) |k| {
            const out_idx = blk * NvFp4Buffers.BLOCK_SIZE + 2 * k;
            const in_idx = blk * NvFp4Buffers.BLOCK_SIZE / 2 + k;
            const byte: u8 = input.fp4block16[in_idx];

            const low: u4 = @as(u4, @truncate(byte & fp4_mask));
            const low_fp: Float4E2M1 = @bitCast(low);
            output[out_idx] = low_fp.toF32() * block_scale * tensor_scale;

            const high: u4 = @as(u4, @truncate((byte >> fp4_size) & fp4_mask));
            const high_fp: Float4E2M1 = @bitCast(high);
            output[out_idx + 1] = high_fp.toF32() * block_scale * tensor_scale;
        }
    }
}

const VectorU8 = @Vector(DEFAULT_SIMD_SIZE / @sizeOf(u8), u8);
const VectorF32 = @Vector(DEFAULT_SIMD_SIZE / @sizeOf(f32), f32);
const VectorU32 = @Vector(DEFAULT_SIMD_SIZE / @sizeOf(u32), u32);

/// Load bytes from adress and repeat them in SIMD vector.
fn load_bytes(comptime nbytes: usize, comptime splat_size: usize, data: [*]const u8) VectorU8 {
    const UintLoad = std.meta.Int(.unsigned, splat_size * 8);
    // Non UB general way of reading
    // TODO we are always aligned
    var val: UintLoad align(DEFAULT_ALIGN) = 0;
    @memcpy(std.mem.asBytes(&val)[0..nbytes], data[0..nbytes]);
    const out: @Vector(DEFAULT_SIMD_SIZE / @sizeOf(UintLoad), UintLoad) = @splat(val);
    return @bitCast(out);
}

/// Convert fp4 to fp32.
///
/// Assumes the same packed payload in all 32 bit lane.
///
/// Inspired by SSE fp16 to fp32 by Fabian "ryg" Giesen.
/// Does not need to handle inf/nan.
/// https://gist.github.com/rygorous/2144712
///
/// Note: For F4E2M1 a 16-entry table lookup is possible as well but zig @suffle
/// does not handle runtime indices (F4E2M1 as lookup codes).
/// Paritularily on Neon, SIMD lookup tables can span multiple registers.
/// On x86, BMI2/AVX512-VBMI2 could also be a potential direction.
inline fn dequant_fp4(repeated: VectorU32) VectorF32 {
    const f32_mantissa = comptime @bitSizeOf(@FieldType(Float32, "mantissa"));
    const f32_exp = comptime @bitSizeOf(@FieldType(Float32, "exponent"));
    const f4_mantissa = comptime @bitSizeOf(@FieldType(Float4E2M1, "mantissa"));
    const f4_exp = comptime @bitSizeOf(@FieldType(Float4E2M1, "exponent"));
    const mantissa_diff = comptime f32_mantissa - f4_mantissa;
    const exp_mantissa_mask = comptime lsbMask(u32, @bitSizeOf(u4) - 1) << mantissa_diff;
    const f32_bias = comptime 127;
    const f4_bias = comptime 1;
    const bias_diff = comptime f32_bias - f4_bias;
    const rebias_factor: f32 = comptime @bitCast(@as(u32, (f32_bias + bias_diff) << f32_mantissa));

    // Shifts to aligned repeated packed values to the most signigicant bit in a u32
    const lshifts = VectorU32{
        @bitSizeOf(f32) - 1 * @bitSizeOf(u4),
        @bitSizeOf(f32) - 2 * @bitSizeOf(u4),
        @bitSizeOf(f32) - 3 * @bitSizeOf(u4),
        @bitSizeOf(f32) - 4 * @bitSizeOf(u4),
    };
    // Shifts to align back the exponent/mantissa boundary to that of f32
    const rshifts: VectorU32 = @splat(f32_exp - f4_exp);
    // Mask to extract the sign bit from mmost significant bit
    const f32_sign_mask: VectorU32 = @splat(1 << (@bitSizeOf(f32) - 1));
    // Multiplicative factor to rebias as f32
    const rebias_vec: VectorF32 = @splat(rebias_factor);
    // Mask for the exponent and mantissa in their target position
    const exp_mantissa_mask_vec: VectorU32 = @splat(exp_mantissa_mask);

    const lshifted = repeated << lshifts;
    const signs = lshifted & f32_sign_mask;
    const aligned = lshifted >> rshifts;
    const masked = aligned & exp_mantissa_mask_vec;
    const signed = masked | signs;
    const rebias = @as(VectorF32, @bitCast(signed)) * rebias_vec;
    return rebias;
}

pub fn dequantF32Simd(input: NvFp4Buffers, output: []align(DEFAULT_ALIGN) f32) void {
    std.debug.assert(output.len == input.nelements);
    const block_size = comptime NvFp4Buffers.BLOCK_SIZE;
    const block_bytes = comptime block_size / 2;
    const f32_per_simd = comptime DEFAULT_SIMD_SIZE / @sizeOf(f32);
    const simd_per_block = comptime block_size / f32_per_simd;
    const fp4_bit_size = comptime 4;
    const byte_read_per_simd = comptime f32_per_simd * fp4_bit_size / 8;

    const tensor_scale: VectorF32 = @splat(input.tensor_scale);

    for (0..input.nBlocks()) |blk| {
        // Hypothesis: we do not dequant the F8E4M3 with SIMD because the
        // SIMD registers are plenty busy with the f4E2M1 while the regular
        // registers are fgree.
        const block_scale_packed: Float8E4M3 = @bitCast(input.block_scale[blk]);
        const block_scale: VectorF32 = @splat(block_scale_packed.toF32());

        inline for (0..simd_per_block) |k| {
            const out_idx = blk * NvFp4Buffers.BLOCK_SIZE + f32_per_simd * k;
            const in_idx = blk * block_bytes + k * byte_read_per_simd;
            const bytes = load_bytes(
                byte_read_per_simd,
                @sizeOf(f32),
                input.fp4block16 + in_idx,
            );
            const floats = dequant_fp4(@bitCast(bytes)) * block_scale * tensor_scale;

            std.debug.assert((out_idx * @sizeOf(f32)) % DEFAULT_ALIGN == 0);
            const out: *align(DEFAULT_ALIGN) VectorF32 = @ptrCast(@alignCast(output.ptr + out_idx));
            out.* = floats;
        }
    }
}

pub fn streamDequantF32(
    fp4block16_reader: *std.Io.Reader,
    block_scale_reader: *std.Io.Reader,
    tensor_scale: f32,
    out_writer: *std.Io.Writer,
    buffer: []align(DEFAULT_ALIGN) u8,
) !void {
    const alignm = comptime std.mem.Alignment.fromByteUnits(DEFAULT_ALIGN);
    const block_size = comptime NvFp4Buffers.BLOCK_SIZE;
    const fp4_block_bytes = comptime block_size / 2;
    // Number of fp4 + fp8 + fp32 blocks we can fit in the buffer
    const buf_block_count = buffer.len / (1 + fp4_block_bytes + @sizeOf(f32) * block_size);
    std.debug.assert(buf_block_count > 0);
    var fp32_buf = buffer[0 .. @sizeOf(f32) * block_size * buf_block_count];
    var fp4_buf = buffer[fp32_buf.len..][0 .. fp4_block_bytes * buf_block_count];
    var fp8_buf = buffer[fp32_buf.len + fp4_buf.len ..][0..buf_block_count];

    var done: usize = 0;
    while (true) {
        const nbfp4 = try fp4block16_reader.readSliceShort(fp4_buf);
        const nbfp8 = try block_scale_reader.readSliceShort(fp8_buf);
        if (nbfp4 != nbfp8 * fp4_block_bytes) {
            @branchHint(.unlikely);
            return error.InvalidTensorShapes;
        }
        if (nbfp4 == 0) {
            break;
        }

        // Prepare input
        const nelements = nbfp8 * block_size;
        const fp4: []align(DEFAULT_ALIGN) u8 = @alignCast(fp4_buf[0..nbfp4]);
        std.debug.assertAligned(fp4, alignm);
        const fp8 = fp8_buf[0..nbfp8];
        const out = fp32_buf[0 .. nelements * @sizeOf(f32)];
        // We do not need to validate again
        std.debug.assert(!std.meta.isError(NvFp4Buffers.validate(fp4, fp8, tensor_scale)));
        const input = NvFp4Buffers{
            .nelements = nelements,
            .fp4block16 = fp4.ptr,
            .block_scale = fp8.ptr,
            .tensor_scale = tensor_scale,
        };
        dequantF32Simd(input, @ptrCast(out));
        done += input.nelements;
        try out_writer.writeAll(out);
    }
    std.debug.print("  Dequant {d} values\n", .{done});
}

fn nibble_swap(x: u8) u8 {
    return (x << 4) | (x >> 4);
}

test nibble_swap {
    try std.testing.expectEqual(nibble_swap(0b00000000), 0b00000000);
    try std.testing.expectEqual(nibble_swap(0b11111111), 0b11111111);
    try std.testing.expectEqual(nibble_swap(0b00001111), 0b11110000);
    try std.testing.expectEqual(nibble_swap(0b10010110), 0b01101001);
}

test "dequantF32" {

    // All F4E2M1 and their F32 vaules
    const block align(DEFAULT_ALIGN) = [_]u8{
        0b0001_0000,
        0b0011_0010,
        0b0101_0100,
        0b0111_0110,
        0b1001_1000,
        0b1011_1010,
        0b1101_1100,
        0b1111_1110,
    };
    const blockf32 = [_]f32{ 0.0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6 };
    // Some F8E4M3 block sclaes and their F32 values
    const scales = [_]u8{ 0b00000000, 0b00110000, 0b11001011 };
    const scalesf32 = [_]f32{ 0.0, 0.5, -5.5 };
    const nelements = scalesf32.len * blockf32.len;

    // Create test fp4 data with three blocks
    var test_fp4: [block.len * scales.len]u8 align(DEFAULT_ALIGN) = undefined;
    for (0..block.len) |k| {
        // Forward
        test_fp4[0 * block.len + k] = block[k];
        // Backward
        test_fp4[1 * block.len + k] = nibble_swap(block[block.len - 1 - k]);
        // Repeated
        test_fp4[2 * block.len + k] = block[k / 2];
    }

    inline for (.{ dequantF32Scalar, dequantF32Simd }) |dequantF32| {
        for (scalesf32) |tscale| {
            var out: [nelements]f32 align(DEFAULT_ALIGN) = undefined;
            @memset(&out, 0.0);

            const input = try NvFp4Buffers.validate(&test_fp4, &scales, tscale);
            dequantF32(input, &out);

            var expected: [nelements]f32 align(DEFAULT_ALIGN) = undefined;
            for (0..blockf32.len) |k| {
                // Forward
                expected[0 * blockf32.len + k] = blockf32[k] * scalesf32[0] * tscale;
                // Backward
                expected[1 * blockf32.len + k] = blockf32[blockf32.len - 1 - k] * scalesf32[1] * tscale;
                // Repeated
                expected[2 * blockf32.len + k] = blockf32[2 * (k / 4) + k % 2] * scalesf32[2] * tscale;
            }

            try std.testing.expectEqualSlices(f32, &expected, &out);
        }
    }
}
