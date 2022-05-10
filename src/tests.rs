use crate::tensor::*;

#[test]
fn test_1d_tensor() {
    let tmp = [3.,2.,4.];
    let tensor: Tensor<3> = Tensor::from_arr::<f64>(&tmp);

    let tmp = [3.,2., 3.];
    let tensor2: Tensor<3> = Tensor::from_arr::<f64>(&tmp);

    let _tensor3 = &tensor + &tensor2;
    let _tensor4 = &tensor - &tensor2;
    let _tensor5 = &tensor * &tensor2;
    let _tensor6 = &tensor / &tensor2;
}

#[test]
fn test_2d_tensor() {
    let tmp = [[3.,2.,4.],[3.4,4.5,2.3]];
    let tensor: Tensor<2, 3> = Tensor::from_arr_2d(&tmp);

    let tmp = [[3.,2., 3.],[12.3,4.4,56.4]];
    let tensor2: Tensor<2, 3> = Tensor::from_arr_2d(&tmp);

    let _tensor3 = &tensor + &tensor2;
    let _tensor4 = &tensor - &tensor2;
    let _tensor5 = &tensor * &tensor2;
    let _tensor6 = &tensor / &tensor2;
}

#[test]
fn add_tensors_fn() {
    fn add<const D1: u16, const D2: u16, const D3: u16, const D4: u16, const D5: u16>(t1: &Tensor<D1, D2, D3, D4, D5>, t2: &Tensor<D1, D2, D3, D4, D5>) -> Tensor<D1, D2, D3, D4, D5>  {
        t1 + t2
    }

    let tmp = [[3.,2.,4.],[3.4,4.5,2.3]];
    let tensor: Tensor<2, 3> = Tensor::from_arr_2d(&tmp);

    let tmp = [[3.,2., 3.],[12.3,4.4,56.4]];
    let tensor2: Tensor<2, 3> = Tensor::from_arr_2d(&tmp);

    let _tensor3 = add(&tensor, &tensor2);
}