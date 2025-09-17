pragma solidity ^0.8.0;
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

// Minimal ReentrancyGuard stub (actual OZ version is more complex)
abstract contract ReentrancyGuard {
    uint256 private constant _NOT_ENTERED = 1;
    uint256 private constant _ENTERED = 2;
    uint256 private _status;
    constructor () {
        _status = _NOT_ENTERED;
    }
    modifier nonReentrant() {
        require(_status != _ENTERED, "ReentrancyGuard: reentrant call");
        _status = _ENTERED;
        _;
        _status = _NOT_ENTERED;
    }
}

// Minimal Ownable stub (actual OZ version is more complex)
abstract contract Ownable {
    address private _owner;
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
    constructor () {
        _owner = msg.sender;
        emit OwnershipTransferred(address(0), msg.sender);
    }
    function owner() public view virtual returns (address) {
        return _owner;
    }
    modifier onlyOwner() {
        require(owner() == msg.sender, "Ownable: caller is not the owner");
        _;
    }
    function renounceOwnership() public virtual onlyOwner {
        emit OwnershipTransferred(_owner, address(0));
        _owner = address(0);
    }
}

/**
 * @title MySafeContract
 * @dev A sample safe contract with NatSpec.
 */
contract MySafeContract is ReentrancyGuard, Ownable {
    uint256 public value;

    /**
     * @dev Sets the value.
     * @param _newValue The new value to set.
     */
    function setValue(uint256 _newValue) public onlyOwner nonReentrant {
        value = _newValue;
    }
}