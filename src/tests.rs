use std::ops::Add;

use crate::tensor::*;

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