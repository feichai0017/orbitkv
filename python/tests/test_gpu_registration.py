from .unit_stubs import install_connector_unit_stubs

install_connector_unit_stubs()

from orbitkv.client import gpu  # noqa: E402

CudaIPCWrapper = gpu.CudaIPCWrapper


class FakeTensor:
    def __init__(self):
        self.set_args = None

    def set_(self, *args):
        self.set_args = args
        return self


class FakeUntypedStorage:
    @staticmethod
    def _new_shared_cuda(device, *handle_args):
        return ("storage", device, handle_args)


class FakeTorch:
    UntypedStorage = FakeUntypedStorage

    def __init__(self):
        self.created_tensor = FakeTensor()

    def tensor(self, *args, **kwargs):
        return self.created_tensor


def _wrapper_without_init(stride, storage_offset):
    wrapper = CudaIPCWrapper.__new__(CudaIPCWrapper)
    wrapper.handle = ("ignored_device", "ipc_handle", "size")
    wrapper.dtype = "fake-dtype"
    wrapper.shape = (2, 3)
    wrapper.device_uuid = "GPU-fake"
    wrapper.stride = stride
    wrapper.storage_offset = storage_offset
    return wrapper


def test_to_tensor_preserves_strided_storage(monkeypatch):
    fake_torch = FakeTorch()
    monkeypatch.setattr(gpu, "torch", fake_torch)
    monkeypatch.setattr(CudaIPCWrapper, "_get_device_index_from_uuid", staticmethod(lambda _: 7))

    tensor = _wrapper_without_init(stride=(4, 1), storage_offset=5).to_tensor()

    assert tensor.set_args == (("storage", 7, ("ipc_handle", "size")), 5, (2, 3), (4, 1))
