const std = @import("std");

const zml = @import("zml");
const dquant = @import("dequant.zig");

const SafeTensor = zml.safetensors.Tensor;
const TensorRegistry = zml.safetensors.TensorRegistry;
const DataType = zml.DataType;
const DEFAULT_ALIGN = dquant.DEFAULT_ALIGN;

const NvFp4SafeTensor = struct {
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

    fn make(
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

    fn fp4BlockByteSize(self: *const Self) usize {
        return self.fp4block16.byteSize();
    }

    fn blockScaleByteSize(self: *const Self) usize {
        return self.block_scale.byteSize();
    }

    fn dequantByteSize(self: *const Self) usize {
        return 2 * self.fp4BlockByteSize();
    }
};

fn tensorDequantF32Scalar(
    tensor: NvFp4SafeTensor,
    allocator: std.mem.Allocator,
    io: std.Io,
) ![]f32 {
    const alignm = comptime std.mem.Alignment.fromByteUnits(DEFAULT_ALIGN);

    // Allocate output before input since the latter can be freed
    const out = try allocator.alignedAlloc(f32, alignm, tensor.dequantByteSize());

    var buffer: [8 * 1024]u8 = undefined;

    // Read fp4 weights
    var fp4block_reader = try tensor.fp4block16.reader(io, &buffer, .{}); // TODO alignment?
    defer fp4block_reader.deinit();
    const fp4block = try allocator.alignedAlloc(u8, alignm, tensor.fp4BlockByteSize());
    try fp4block_reader.interface.readSliceAll(fp4block);
    defer allocator.free(fp4block);

    // Read fp8 scales
    var bscale_reader = try tensor.block_scale.reader(io, &buffer, .{}); // TODO alignment?
    defer bscale_reader.deinit();
    const block_scale = try allocator.alignedAlloc(u8, alignm, tensor.blockScaleByteSize());
    try bscale_reader.interface.readSliceAll(block_scale);
    defer allocator.free(block_scale);

    // Read tensor scale
    var tscale_reader = try tensor.tensor_scale.reader(io, &buffer, .{});
    defer tscale_reader.deinit();
    var tensor_scale: f32 = undefined;
    try tscale_reader.interface.readSliceAll(std.mem.asBytes(&tensor_scale));

    // Dequant
    const input = try dquant.NvFp4Buffers.validate(fp4block, block_scale, tensor_scale);
    dquant.dequantF32Scalar(input, out);

    return out;
}

// Notes:
// We could improve on the reader to void full allocation
// We did not account for endianness
pub fn main(init: std.process.Init) !void {
    const arena = init.arena;
    const allocator = init.gpa;
    const io = init.io;

    var threaded: std.Io.Threaded = .init(allocator, .{});
    defer threaded.deinit();

    // Parse program args
    const process_args = try init.minimal.args.toSlice(arena.allocator());
    const model_path = process_args[1];
    std.debug.print("Reading, {s}\n", .{model_path});

    // Read model shapes.
    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, model_path);
    defer registry.deinit();

    var it = registry.iterator();
    while (it.next()) |t| {
        const tensor = t.value_ptr;
        const nvfp4t = try NvFp4SafeTensor.make(tensor, &registry, allocator);
        if (nvfp4t != null) {
            const buff32 = try tensorDequantF32Scalar(nvfp4t.?, allocator, io);
            defer allocator.free(buff32);
        }
    }
}

test {
    _ = dquant;
}
