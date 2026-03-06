
import os
import torch
import time

os.environ["MOONCAKE_MASTER"] = "s-sglang-demo-mooncake-master:50051"
os.environ["MOONCAKE_TE_META_DATA_SERVER"] = "http://s-sglang-demo-mooncake-master:8080/metadata"
os.environ["MOONCAKE_GLOBAL_SEGMENT_SIZE"] = "20gb"
os.environ["MOONCAKE_PROTOCOL"] = "rdma"
os.environ["MC_GID_INDEX"] = "9"

DEFAULT_LOCAL_BUFFER_SIZE = 16 * 1024 * 1024
from mooncake.store import MooncakeDistributedStore, ReplicateConfig

# try:
#     from mooncake.store import MooncakeDistributedStore
# except ImportError as e:
#     raise ImportError(
#         "Please install mooncake by following the instructions at "
#         "https://kvcache-ai.github.io/Mooncake/getting_started/build.html"
#         "to run SGLang with MooncakeConnector."
#     ) from e

store = MooncakeDistributedStore()

from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStoreConfig
config = MooncakeStoreConfig.load_from_env()

per_tp_global_segment_size = config.global_segment_size

# temporarily set to empty
device_name=""

store.setup(
    config.local_hostname,
    config.metadata_server,
    per_tp_global_segment_size,
    DEFAULT_LOCAL_BUFFER_SIZE,  # Zero copy interface does not need local buffer
    config.protocol,
    device_name,
    config.master_server_address,
)

rep_config = ReplicateConfig()
rep_config.replica_num = 1             # Number of replicas
rep_config.preferred_segment = "localhost:13079"

mytensor = torch.rand((1024, 1024, 1024), device="cuda")
print(mytensor.dtype)
tensor_size = mytensor.untyped_storage().nbytes()
print(tensor_size)
tensor_ptr = mytensor.data_ptr()

ret_code = store.register_buffer(tensor_ptr, tensor_size)
if ret_code:
    raise RuntimeError(
        f"Failed to register buffer to Mooncake Store, error code: {ret_code}"
    )

start_time = time.time()

store.batch_put_from(["mytensor-202603061455"], [tensor_ptr], [tensor_size], rep_config)

print(f"Put time: {time.time() - start_time}")
# config.with_soft_pin = True         # Keep in memory longer
# config.preferred_segment = "host:port"  # Preferred location

