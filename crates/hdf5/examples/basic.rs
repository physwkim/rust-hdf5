//! Basic usage example demonstrating contiguous, chunked, and compressed datasets.

use hdf5::H5File;
use hdf5::types::VarLenUnicode;

fn main() {
    let path = "/tmp/hdf5rs_example.h5";

    // --- Write ---
    {
        let file = H5File::create(path).unwrap();

        // 1) Contiguous dataset
        let ds = file.new_dataset::<f64>()
            .shape([4usize, 5])
            .create("matrix")
            .unwrap();
        let data: Vec<f64> = (0..20).map(|i| i as f64 * 0.1).collect();
        ds.write_raw(&data).unwrap();

        // 2) Chunked dataset with compression
        let stream = file.new_dataset::<i32>()
            .shape([0usize, 3])
            .chunk(&[1, 3])
            .max_shape(&[None, Some(3)])
            .deflate(6)
            .create("stream")
            .unwrap();

        for frame in 0..100u64 {
            let vals: Vec<i32> = (0..3).map(|i| (frame * 3 + i) as i32).collect();
            let raw: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
            stream.write_chunk(frame as usize, &raw).unwrap();
        }
        stream.extend(&[100, 3]).unwrap();

        // 3) String attribute
        let attr = ds.new_attr::<VarLenUnicode>()
            .shape(())
            .create("units")
            .unwrap();
        attr.write_string("meters").unwrap();

        file.close().unwrap();
        println!("Wrote {}", path);
    }

    // --- Read ---
    {
        let file = H5File::open(path).unwrap();
        println!("Datasets: {:?}", file.dataset_names());

        let ds = file.dataset("matrix").unwrap();
        println!("matrix shape: {:?}", ds.shape());
        let data = ds.read_raw::<f64>().unwrap();
        println!("matrix[0..5]: {:?}", &data[..5]);

        // Slice read
        let slice = ds.read_slice::<f64>(&[1, 2], &[2, 2]).unwrap();
        println!("matrix[1:3, 2:4] = {:?}", slice);

        let stream = file.dataset("stream").unwrap();
        println!("stream shape: {:?}", stream.shape());
        let sdata = stream.read_raw::<i32>().unwrap();
        println!("stream total elements: {}", sdata.len());
    }

    std::fs::remove_file(path).ok();
}
