use std::ffi::CStr;

use cudarc::runtime::result::device;

fn main() {
    let count = device::get_count().unwrap();
    let curr = device::get().unwrap();
    let props = device::get_device_prop(curr).unwrap();

    let name = unsafe { CStr::from_ptr(props.name.as_ptr()).to_str().unwrap() };
    println!("count= {count} name= {name}");
}
