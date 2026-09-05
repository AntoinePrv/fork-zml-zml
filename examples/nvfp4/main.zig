const std = @import("std");

const zml = @import("zml");
const dquant = @import("dequant.zig");
const sfdquant = @import("safetensors.zig");

const SafeTensor = zml.safetensors.Tensor;
const TensorRegistry = zml.safetensors.TensorRegistry;
const DataType = zml.DataType;
const DEFAULT_ALIGN = dquant.DEFAULT_ALIGN;

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
        const nvfp4t = try sfdquant.NvFp4SafeTensor.make(tensor, &registry, allocator);
        if (nvfp4t != null) {
            const buff32 = try sfdquant.tensorDequantF32Scalar(nvfp4t.?, allocator, io);
            std.mem.doNotOptimizeAway(&buff32);
            defer allocator.free(buff32);
        }
    }
}

test {
    _ = dquant;
}
