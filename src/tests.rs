use crate::tensor::*;

#[test]
fn test_1d_tensor() {
    let tmp = [3.,2.,4.];
    let tensor: Tensor<{DimType::Static(3)}> = Tensor::from_arr::<f64>(&tmp);

    let tmp = [3.,2., 3.];
    let tensor2: Tensor<{DimType::Static(3)}> = Tensor::from_arr::<f64>(&tmp);

    let _tensor3 = &tensor + &tensor2;
    let _tensor4 = &tensor - &tensor2;
    let _tensor5 = &tensor * &tensor2;
    let _tensor6 = &tensor / &tensor2;
}

#[test]
fn test_2d_tensor() {
    let tmp = [[3.,2.,4.],[3.4,4.5,2.3]];
    let tensor: Tensor<{DimType::Static(2)}, {DimType::Static(3)}> = Tensor::from_arr_2d(&tmp);

    let tmp = [[3.,2., 3.],[12.3,4.4,56.4]];
    let tensor2: Tensor<{DimType::Static(2)}, {DimType::Static(3)}> = Tensor::from_arr_2d(&tmp);

    let _tensor3 = &tensor + &tensor2;
    let _tensor4 = &tensor - &tensor2;
    let _tensor5 = &tensor * &tensor2;
    let _tensor6 = &tensor / &tensor2;
}

#[test]
fn add_tensors_fn() {
    fn add<const DIM1: DimType, const DIM2: DimType, const DIM3: DimType, const DIM4: DimType, const DIM5: DimType>(t1: &Tensor<DIM1, DIM2, DIM3, DIM4, DIM5>, t2: &Tensor<DIM1, DIM2, DIM3, DIM4, DIM5>) -> Tensor<DIM1, DIM2, DIM3, DIM4, DIM5>  {
        t1 + t2
    }

    let tmp = [[3.,2.,4.],[3.4,4.5,2.3]];
    let tensor: Tensor<{DimType::Static(2)}, {DimType::Static(3)}> = Tensor::from_arr_2d(&tmp);

    let tmp = [[3.,2., 3.],[12.3,4.4,56.4]];
    let tensor2: Tensor<{DimType::Static(2)}, {DimType::Static(3)}> = Tensor::from_arr_2d(&tmp);

    let _tensor3 = add(&tensor, &tensor2);
}