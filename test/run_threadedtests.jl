PerformanceTestTools.@include_foreach(
    "test_threaded.jl",
    [nothing,
    ["JULIA_NUM_THREADS" => Threads.nthreads() > 1 ? "1" : "2", "MKL_NUM_THREADS" => "3"],
    ],
)

