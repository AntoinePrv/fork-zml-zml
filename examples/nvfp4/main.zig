const std = @import("std");

const zml = @import("zml");
const dquant = @import("dequant.zig");
const sfdquant = @import("safetensors.zig");

const SafeTensor = zml.safetensors.Tensor;
const TensorRegistry = zml.safetensors.TensorRegistry;
const DataType = zml.DataType;
const DEFAULT_ALIGN = dquant.DEFAULT_ALIGN;

fn dequantThread(
    registry: *const zml.safetensors.TensorRegistry,
    iter: *zml.safetensors.Tensors.Iterator,
    lock: *std.Io.RwLock,
    allocator: std.mem.Allocator,
    io: std.Io,
    id: usize,
    method: sfdquant.Method,
) !void {
    var tmp = std.heap.ArenaAllocator.init(allocator);
    defer tmp.deinit();

    try lock.lock(io);
    var locked: bool = true;
    defer if (locked) lock.unlock(io);

    while (iter.next()) |t| {
        const tensor = t.value_ptr;
        const nvfp4t = try sfdquant.NvFp4SafeTensor.make(tensor, registry, tmp.allocator());
        if (nvfp4t != null) {
            // Let other thread find their work
            locked = false;
            lock.unlock(io);

            std.debug.print(
                \\  [Thread:{d}] Reading {s}
                \\      - shape: {f}
                \\      - elements: {d}
                \\
            ,
                .{ id, tensor.name, tensor.shape, nvfp4t.?.nelements() },
            );
            const buff32 = try sfdquant.tensorDequantF32(
                nvfp4t.?,
                allocator,
                tmp.allocator(),
                io,
                method,
            );

            // Fake usage of the data
            std.mem.doNotOptimizeAway(&buff32);
            defer allocator.free(buff32);

            try lock.lock(io);
            locked = true;
        }
    }
}

// Notes:
//
// - The Reader/Writer design may addssome overhead which seems to mitigate SIMD improvements.
//   Investigate overlapping IO and compute.
// - TODO: investigate a SIMD lookup algorithm:
//     Shuffle dyn would be good to make lookup into fp32 values (since there are few).
//     vqtbl1q_u8 etc on Neon and vqtbl4q_u8 for multi register table
//     Also BMI2 and VMBI
//     Though Zig @shuffle does not support dynamic masks. An inline for loop my be detected
//     by the compiler.
//     https://github.com/ziglang/zig/issues/24810
// - We did not account for endianness
pub fn main(init: std.process.Init) !void {
    const arena = init.arena;
    const allocator = init.gpa;
    const io = init.io;

    var threaded: std.Io.Threaded = .init(allocator, .{});
    defer threaded.deinit();

    // Parse program args
    const process_args = try init.minimal.args.toSlice(arena.allocator());
    const model_path = process_args[1];
    const method = try sfdquant.Method.parse(process_args[2]);
    const nthreads = try std.fmt.parseInt(usize, process_args[3], 10);

    std.debug.print("Reading, {s}\n", .{model_path});

    // Read model shapes.
    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, model_path);
    defer registry.deinit();

    // Thread pool
    var threads = try std.ArrayList(std.Thread).initCapacity(allocator, 16);
    defer threads.deinit(allocator);

    // Launching threads with their
    var iter = registry.iterator();
    var lock = std.Io.RwLock.init;
    std.debug.print(
        "Launching dequantization with {s} on {d} threads\n",
        .{ @tagName(method), nthreads },
    );
    for (0..nthreads - 1) |id| {
        const t = try std.Thread.spawn(
            .{},
            dequantThread,
            .{ &registry, &iter, &lock, allocator, io, id + 1, method },
        );
        try threads.append(allocator, t);
    }
    // Main thread pulls some work too
    const result = dequantThread(&registry, &iter, &lock, allocator, io, 0, method);

    // Check all errors
    try result;
    for (threads.items) |t| {
        t.join();
    }
}

test {
    _ = dquant;
}
