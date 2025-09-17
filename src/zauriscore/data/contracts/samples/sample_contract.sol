// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";

/**
 * @title SafeTokenExample
 * @dev Demonstrates best practices in smart contract security
 */
contract SafeTokenExample is ERC20, Ownable, ReentrancyGuard {
    // Maximum total supply
    uint256 private constant MAX_SUPPLY = 1_000_000 * 10**18;
    
    // Transfer fee parameters
    uint256 public constant TRANSFER_FEE_PERCENT = 1; // 1% transfer fee
    address public feeCollector;

    // Blacklist for suspicious addresses
    mapping(address => bool) public blacklisted;

    // Events for enhanced transparency
    event FeeCollectorUpdated(address indexed newFeeCollector);
    event AddressBlacklisted(address indexed user, bool status);

    constructor() ERC20("SafeToken", "SAFE") {
        // Mint initial supply to contract creator
        _mint(msg.sender, MAX_SUPPLY);
        feeCollector = msg.sender;
    }

    /**
     * @dev Override transfer to add fee mechanism and blacklist check
     */
    function transfer(address recipient, uint256 amount) 
        public 
        virtual 
        override 
        nonReentrant 
        returns (bool) 
    {
        // Check blacklist
        require(!blacklisted[msg.sender] && !blacklisted[recipient], "Blacklisted address");

        // Calculate transfer fee
        uint256 fee = (amount * TRANSFER_FEE_PERCENT) / 100;
        uint256 amountAfterFee = amount - fee;

        // Perform transfer with fee
        super.transfer(recipient, amountAfterFee);
        if (fee > 0) {
            super.transfer(feeCollector, fee);
        }

        return true;
    }

    /**
     * @dev Allows owner to update fee collector address
     */
    function updateFeeCollector(address _newFeeCollector) 
        external 
        onlyOwner 
    {
        require(_newFeeCollector != address(0), "Invalid address");
        feeCollector = _newFeeCollector;
        emit FeeCollectorUpdated(_newFeeCollector);
    }

    /**
     * @dev Allows owner to blacklist/unblacklist addresses
     */
    function setBlacklist(address _user, bool _status) 
        external 
        onlyOwner 
    {
        blacklisted[_user] = _status;
        emit AddressBlacklisted(_user, _status);
    }

    /**
     * @dev Prevents minting beyond max supply
     */
    function mint(address to, uint256 amount) 
        external 
        onlyOwner 
    {
        require(
            totalSupply() + amount <= MAX_SUPPLY, 
            "Exceeds maximum supply"
        );
        _mint(to, amount);
    }

    /**
     * @dev Emergency stop mechanism
     */
    function emergencyWithdraw() 
        external 
        onlyOwner 
        nonReentrant 
    {
        uint256 balance = address(this).balance;
        payable(owner()).transfer(balance);
    }
}
