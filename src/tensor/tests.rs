use std::ops::Add;

use crate::*;

#[test]
fn test_1d_tensor() {
    let tmp = [3.,2.,4.];
    let tensor: Tensor1<3> = Tensor1::from_arr(&tmp);

    let tmp = [3.,2., 3.];
    let tensor2: Tensor1<3> = Tensor1::from_arr(&tmp);

    let _tensor3 = &tensor + &tensor2;
    let _tensor4 = &tensor - &tensor2;
    let _tensor5 = &tensor * &tensor2;
    let _tensor6 = &tensor / &tensor2;
}

#[test]
fn test_2d_tensor() {
    let tmp = [[3.,2.,4.],[3.4,4.5,2.3]];
    let tensor: Tensor2<2, 3> = Tensor2::from_arr(&tmp);

    let tmp = [[3.,2., 3.],[12.3,4.4,56.4]];
    let tensor2: Tensor2<2, 3> = Tensor2::from_arr(&tmp);

    let _tensor3 = &tensor + &tensor2;
    let _tensor4 = &tensor - &tensor2;
    let _tensor5 = &tensor * &tensor2;
    let _tensor6 = &tensor / &tensor2;
}

#[test]
fn test_reshape() {
    fn unsqueeze_repeat_tensor2<const D1: u16, const D2: u16, const REPEATS: u16>(tensor2: Tensor3<1, D1, D2>) -> Tensor3<REPEATS, D1, D2> {
        tensor2.repeat_1()
    }
    const BATCH: u16 = 10;
    let tensor: Tensor2<2, 3> = Tensor2::rand();
    //let tensor4: Tensor2< = tensor.repeat_2();
    let tensor1 = tensor.unsqueeze_start();
    let _tensor2: Tensor3<BATCH, 2, 3> = unsqueeze_repeat_tensor2(tensor1);
}

#[test]
fn add_tensors_fn() {
    fn add<A: Add<B>, B>(t1: A, t2: B) -> <A as std::ops::Add<B>>::Output  {
        t1 + t2
    }

    let tmp = [[3.,2.,4.],[3.4,4.5,2.3]];
    let tensor: Tensor2<2, 3> = Tensor2::from_arr(&tmp);

    let tmp = [[3.,2., 3.],[12.3,4.4,56.4]];
    let tensor2: Tensor2<2, 3> = Tensor2::from_arr(&tmp);

    let _tensor3 = add(&tensor, &tensor2);
}