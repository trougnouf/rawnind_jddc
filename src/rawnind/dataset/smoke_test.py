# smoke_test.py
import trio

from rawnind.dataset.manager import DatasetIndex


async def smoke_test():
    """Live test with actual RawNIND dataset."""
    index = DatasetIndex(dataset_root="tmp/rawnind_dataset")
    index.discover()
    print("\nInitial state:")
    index.print_summary()

    print("\nDiscovering local files...")
    found, total = index.discover()
    print(f"Found {found}/{total} files")

    successful = 0
    failed = 0
    count = 0

    async for img_info in index.produce_missing_files():
        if count >= 3:  # Limit to 3 photos for smoke test
            break

        try:
            await index._download_file(img_info.download_url, img_info.local_path)
            successful += 1
            print(f"  Downloaded: {img_info.filename}")
        except Exception as e:
            print(f"  Failed: {img_info.filename} - {e}")
            failed += 1

        count += 1
    if count == 0:
        print("No files needed downloading.")

    print(f"\nDownloaded: {successful} successful, {failed} failed")

    print("\nFinal state:")
    index.print_summary()


if __name__ == "__main__":
    trio.run(smoke_test)
