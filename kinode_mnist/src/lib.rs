//use image::Pixel;

use kinode_process_lib::{await_message, call_init, println, Address, LazyLoadBlob, Message, Request, Response};

mod ml_types;

use ml_types::{KinodeMlLibrary, KinodeMlDataType, KinodeMlRequest};

wit_bindgen::generate!({
    path: "wit",
    world: "process",
    exports: {
        world: Component,
    },
});

const MODEL: &[u8] = include_bytes!("./TFKeras.h5");
const DATA: &[u8] = include_bytes!("./test3.png");

fn handle_message(_our: &Address) -> anyhow::Result<()> {
    let message = await_message()?;

    match message {
        Message::Response { .. } => {
            return Err(anyhow::anyhow!("unexpected Response: {:?}", message));
        }
        Message::Request {
            ref body,
            ..
        } => {
            let body: serde_json::Value = serde_json::from_slice(body)?;
            println!("kinode_mnist: got {body:?}");
            Response::new()
                .body(serde_json::to_vec(&serde_json::json!("Ack")).unwrap())
                .send()
                .unwrap();
        }
    }
    Ok(())
}

call_init!(init);

fn init(our: Address) {
    println!("kinode_mnist: begin");

    let data = image::load_from_memory_with_format(DATA, image::ImageFormat::Png).unwrap();
    let data = data.to_luma8();
    println!("{:?}", data);
    //let data = match data {
    //    image::DynamicImage::ImageLuma8(data) => data,
    //    image::DynamicImage::ImageRgb8(data) => data.convert(),
    //    _ => panic!("a"),
    //};
    let data = data
        .into_raw()
        .iter()
        .map(|p| *p as f32 / 255.0)
        .flat_map(|p| p.to_ne_bytes().to_vec())
        .collect();

    Request::new()
        .target("our@ml:ml:sys".parse::<Address>().unwrap())
        .body(serde_json::to_vec(&serde_json::json!("Run")).unwrap())
        .blob(LazyLoadBlob {
            mime: None,
            bytes: rmp_serde::to_vec_named(&KinodeMlRequest {
                library: KinodeMlLibrary::Keras,
                data_shape: vec![1, 784],
                data_type: KinodeMlDataType::Float32,
                model_bytes: MODEL.to_vec(),
                data_bytes: data,
            }).unwrap(),
        })
        .expects_response(15)
        //.inherit(true)
        .send()
        .unwrap();

    loop {
        match handle_message(&our) {
            Ok(()) => {}
            Err(e) => {
                println!("kinode_mnist: error: {:?}", e);
            }
        };
    }
}
