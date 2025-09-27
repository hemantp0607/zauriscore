// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract TestContract {
    uint public value;
    
    function setValue(uint _value) public {
        value = _value;
    }
}
