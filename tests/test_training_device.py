import unittest

from training_device import resolve_device_name


class TrainingDeviceTests(unittest.TestCase):
    def test_cuda_request_fails_instead_of_silently_using_cpu(self):
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            resolve_device_name(
                requested_device="cuda",
                cuda_available=False,
            )

    def test_cuda_request_uses_cuda_when_available(self):
        self.assertEqual(
            resolve_device_name(
                requested_device="cuda",
                cuda_available=True,
            ),
            "cuda",
        )

    def test_explicit_cpu_request_is_allowed(self):
        self.assertEqual(
            resolve_device_name(
                requested_device="cpu",
                cuda_available=False,
            ),
            "cpu",
        )


if __name__ == "__main__":
    unittest.main()
