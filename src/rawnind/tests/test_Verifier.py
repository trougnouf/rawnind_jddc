import pytest
import trio

from rawnind.dataset.Verifier import Verifier


@pytest.mark.asyncio
async def test_consume_new_files_verified_file(tmp_path):
    recv_channel, verified_send = trio.open_memory_channel(0)
    _, missing_send = trio.open_memory_channel(0)

    # Create a temporary file and compute its hash
    temp_file = tmp_path / "test_file.txt"
    temp_file.write_text("Test content")
    expected_hash = Verifier.hash_sha1(temp_file)

    # Create ImageInfo instance with the computed hash
    class ImageInfo:
        def __init__(self, local_path, sha1, filename):
            self.local_path = local_path
            self.sha1 = sha1
            self.filename = filename
            self.validated = False

    img_info = ImageInfo(local_path=temp_file, sha1=expected_hash, filename="test_file.txt")

    # Instantiate Verifier and run the consume_new_files method
    verifier = Verifier()
    await verifier.consume_new_files(recv_channel, verified_send, missing_send)

    # Send img_info through recv_channel
    await recv_channel.send(img_info)

    # Await received message from verified_send channel
    received_img_info = await verified_send.receive()

    assert received_img_info.validated is True


@pytest.mark.asyncio
async def test_consume_new_files_missing_file():
    recv_channel, verified_send = trio.open_memory_channel(0)
    _, missing_send = trio.open_memory_channel(0)

    # Create ImageInfo instance with None local_path
    class ImageInfo:
        def __init__(self, local_path, sha1, filename):
            self.local_path = local_path
            self.sha1 = sha1
            self.filename = filename
            self.validated = False

    img_info = ImageInfo(local_path=None, sha1="d41d8cd98f00b204e9800998ecf8427e", filename="missing_file.txt")

    # Instantiate Verifier and run the consume_new_files method
    verifier = Verifier()
    await verifier.consume_new_files(recv_channel, verified_send, missing_send)

    # Send img_info through recv_channel
    await recv_channel.send(img_info)

    # Await received message from missing_send channel
    received_img_info = await missing_send.receive()

    assert received_img_info.validated is False
