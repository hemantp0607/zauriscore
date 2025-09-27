// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";
import "@openzeppelin/contracts-upgradeable/access/AccessControlUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/security/ReentrancyGuardUpgradeable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/utils/math/SafeMath.sol";

/**
 * @title SecureUpgradeableToken
 * @dev Demonstrates comprehensive smart contract security practices
 */
contract SecureUpgradeableToken is 
    Initializable, 
    AccessControlUpgradeable, 
    ReentrancyGuardUpgradeable 
{
    using SafeMath for uint256;

    // Roles
    bytes32 public constant MINTER_ROLE = keccak256("MINTER_ROLE");
    bytes32 public constant PAUSER_ROLE = keccak256("PAUSER_ROLE");

    // State variables
    mapping(address => uint256) private _balances;
    uint256 private _totalSupply;
    bool private _paused;

    // Events for transparency
    event TokensMinted(address indexed to, uint256 amount);
    event TokensBurned(address indexed from, uint256 amount);
    event EmergencyPaused(address indexed by);
    event EmergencyUnpaused(address indexed by);

    /**
     * @dev Initializer replaces constructor for upgradeable contracts
     */
    function initialize(address initialAdmin) public initializer {
        __AccessControl_init();
        __ReentrancyGuard_init();

        // Setup roles
        _setupRole(DEFAULT_ADMIN_ROLE, initialAdmin);
        _setupRole(MINTER_ROLE, initialAdmin);
        _setupRole(PAUSER_ROLE, initialAdmin);
    }

    /**
     * @dev Mint tokens with checks-effects-interactions pattern
     */
    function mint(address to, uint256 amount) 
        external 
        nonReentrant 
        onlyRole(MINTER_ROLE) 
    {
        // Checks
        require(to != address(0), "Invalid recipient");
        require(amount > 0, "Amount must be positive");

        // Effects
        _balances[to] = _balances[to].add(amount);
        _totalSupply = _totalSupply.add(amount);

        // Interactions (event emission)
        emit TokensMinted(to, amount);
    }

    /**
     * @dev Burn tokens safely
     */
    function burn(uint256 amount) 
        external 
        nonReentrant 
    {
        // Checks
        require(_balances[msg.sender] >= amount, "Insufficient balance");

        // Effects
        _balances[msg.sender] = _balances[msg.sender].sub(amount);
        _totalSupply = _totalSupply.sub(amount);

        // Interactions
        emit TokensBurned(msg.sender, amount);
    }

    /**
     * @dev Safe external token transfer
     */
    function transferTokens(IERC20 token, address to, uint256 amount) 
        external 
        nonReentrant 
        onlyRole(DEFAULT_ADMIN_ROLE) 
    {
        // Checks
        require(to != address(0), "Invalid recipient");
        require(amount > 0, "Amount must be positive");

        // Interactions with external call
        bool success = token.transfer(to, amount);
        require(success, "Token transfer failed");
    }

    /**
     * @dev Emergency pause mechanism
     */
    function emergencyPause() 
        external 
        onlyRole(PAUSER_ROLE) 
    {
        _paused = true;
        emit EmergencyPaused(msg.sender);
    }

    /**
     * @dev Emergency unpause mechanism
     */
    function emergencyUnpause() 
        external 
        onlyRole(PAUSER_ROLE) 
    {
        _paused = false;
        emit EmergencyUnpaused(msg.sender);
    }

    /**
     * @dev Prevents using tx.origin for authentication
     */
    modifier onlyEOA() {
        require(msg.sender == tx.origin, "Only EOA");
        _;
    }

    /**
     * @dev Fallback function to reject direct sends
     */
    receive() external payable {
        revert("Direct sends not allowed");
    }
}
