use micrograd_rs::data::mnist::{load_images,load_labels};



fn main(){
    let images=load_images("data/mnist/train-images-idx3-ubyte");
    let labels=load_labels("data/mnist/train-labels-idx1-ubyte");
    for r in 0..28{
        for c in 0..28{
            let pixel=images.get(0,r*28 +c);
            if pixel>0.5{
                print!("#");
            }
            else if pixel>0.0{
                print!(".");
            }
            else {
                print!(" ");
            }
        }
        println!("");
    }
    println!("label:{}",labels[0]);
}