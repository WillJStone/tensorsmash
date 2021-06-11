use ndarray::{Array, Dim};


pub fn relu(array: Array<f32, Dim<[usize; 2]>>) -> Array<f32, Dim<[usize; 2]>> {
    fn _relu(x: f32) -> f32 {
        if x > 0.0 {
            return x
        } else {
            return 0.0
        }
    }

    array.mapv_into(|x| _relu(x))
}


pub fn concat(arrays: &[Array<f32, Dim<[usize; 1]>>]) -> Array<f32, Dim<[usize; 1]>> {
    let mut new_array: Vec<f32> = Vec::new();
    
    for array in arrays.iter() {
        for elem in array.iter() {
            new_array.push(*elem);
        }
    }
    
    Array::from(new_array)
}


pub fn reshape_array(array: Array<f32, Dim<[usize; 1]>>, shape: [usize; 2]) -> Array<f32, Dim<[usize; 2]>> {
    let mut new_array: Array<f32, Dim<[usize; 2]>> = Array::zeros(shape);
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            new_array[[i, j]] = array[[i * shape[1] + j]]
        }
    }
    new_array
}


#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_reshape_array() {
        let arr: Array<f32, Dim<[usize; 2]>> = Array::eye(2);
        let flat_arr = Array::from(vec![1., 0., 0., 1.]);
        let reshaped_arr = reshape_array(flat_arr, [2, 2]);
        assert_eq!(reshaped_arr, arr);
    }
}