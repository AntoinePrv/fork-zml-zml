const std = @import("std");

const zml = @import("zml");
const dquant = @import("dequant.zig");

const SafeTensor = zml.safetensors.Tensor;
const TensorRegistry = zml.safetensors.TensorRegistry;
const DataType = zml.DataType;
const DEFAULT_ALIGN = dquant.DEFAULT_ALIGN;

pub const NvFp4SafeTensor = struct {
    const Self = @This();

    const BLOCK_SIZE = 16;
    const BLOCK_DTYPE = DataType.u8;
    const BLOCK_SCALE_SUFFIX = "_scale";
    const BLOCK_SCALE_DTYPE = DataType.f8e4m3fn;
    const TENSOR_SCALE_SUFFIX = "_scale_2";
    const TENSOR_SCALE_DTYPE = DataType.f32;

    fp4block16: SafeTensor,
    block_scale: SafeTensor,
    tensor_scale: SafeTensor,

    pub fn make(
        tensor: *const SafeTensor,
        registry: *const TensorRegistry,
        allocator: std.mem.Allocator,
    ) !?Self {
        const dtype = tensor.*.shape.dtype();
        if (dtype != BLOCK_DTYPE) {
            return null;
        }

        const bs_name = try std.mem.concat(
            allocator,
            u8,
            &[_][]const u8{ tensor.name, BLOCK_SCALE_SUFFIX },
        );
        defer allocator.free(bs_name);
        const ts_name = try std.mem.concat(
            allocator,
            u8,
            &[_][]const u8{ tensor.name, TENSOR_SCALE_SUFFIX },
        );
        defer allocator.free(ts_name);

        const bs_tensor = registry.*.tensors.getPtr(bs_name);
        const ts_tensor = registry.*.tensors.getPtr(ts_name);
        if (bs_tensor != null and ts_tensor != null) {
            if (bs_tensor.?.*.shape.dtype() == BLOCK_SCALE_DTYPE and
                ts_tensor.?.*.shape.dtype() == TENSOR_SCALE_DTYPE)
            {
                return .{
                    .fp4block16 = tensor.*,
                    .block_scale = bs_tensor.?.*,
                    .tensor_scale = ts_tensor.?.*,
                };
            }
        }
        return null;
    }

    pub fn fp4TensorByteSize(self: *const Self) usize {
        return self.fp4block16.byteSize();
    }

    pub fn fp8ScaleTensorByteSize(self: *const Self) usize {
        return self.block_scale.byteSize();
    }

    pub fn nelements(self: *const Self) usize {
        return 2 * self.fp4TensorByteSize();
    }

    pub fn dequantF32ByteSize(self: *const Self) usize {
        return @sizeOf(f32) * self.nelements();
    }
};

pub fn tensorDequantF32(
    tensor: NvFp4SafeTensor,
    allocator: std.mem.Allocator,
    tmp_alloc: std.mem.Allocator,
    io: std.Io,
) ![]f32 {
    const alignm = comptime std.mem.Alignment.fromByteUnits(DEFAULT_ALIGN);

    // Allocate output before input since the latter can be freed
    const out = try allocator.alignedAlloc(f32, alignm, tensor.nelements());

    // No need for buffer we read
    // Reader for fp4 weights
    var fp4block_reader = try tensor.fp4block16.reader(io, &.{}, .{});
    defer fp4block_reader.deinit();
    // Reader for fp8 scales
    var block_scale_reader = try tensor.block_scale.reader(io, &.{}, .{});
    defer block_scale_reader.deinit();

    // Read tensor scale
    var tensor_scale: f32 = undefined;
    var tscale_reader = try tensor.tensor_scale.reader(io, std.mem.asBytes(&tensor_scale), .{});
    defer tscale_reader.deinit();
    try tscale_reader.interface.readSliceAll(std.mem.asBytes(&tensor_scale));

    // Dequant output
    var writer = std.Io.Writer.fixed(std.mem.sliceAsBytes(out));

    // Dequant working buffer
    const buf = try tmp_alloc.alignedAlloc(u8, alignm, (1 + 8 + 64) * 1024);
    defer tmp_alloc.free(buf);

    try dquant.streamDequantF32(
        &fp4block_reader.interface,
        &block_scale_reader.interface,
        tensor_scale,
        &writer,
        buf,
    );

    return out;
}
