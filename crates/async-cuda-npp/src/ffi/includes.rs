use crate::ffi::utils::cpp;

cpp! {{
    #include <cstdint>
}}

cpp! {{
    #include <cuda_runtime.h>
}}

cpp! {{
    #include <nppcore.h>
    #include <nppi.h>
}}
