// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract SmartContractInsurance {

    address public owner;

    modifier onlyOwner() {
        require(msg.sender == owner, "Caller is not the owner");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    enum ContractStatus { Pending, Rated, Insured }
    
    struct SubmittedContract {
        address contractAddress;
        string riskReport;
        bool insuranceOffered;
        bool insuranceAccepted;
        bool monitoringEnabled;
        ContractStatus status;
    }

    mapping(address => SubmittedContract[]) public userContracts;

    event ContractSubmitted(address indexed user, address contractAddress);
    event RiskReportUpdated(address indexed user, uint index, string report);
    event InsuranceAccepted(address indexed user, uint index);
    event MonitoringEnabled(address indexed user, uint index);

    // Submit a smart contract
    function submitContract(address _contractAddress) external {
        SubmittedContract memory newSubmission = SubmittedContract({
            contractAddress: _contractAddress,
            riskReport: "",
            insuranceOffered: false,
            insuranceAccepted: false,
            monitoringEnabled: false,
            status: ContractStatus.Pending
        });
        userContracts[msg.sender].push(newSubmission);
        emit ContractSubmitted(msg.sender, _contractAddress);
    }

    // Update risk report (admin-like functionality, can be restricted)
    function updateRiskReport(address _user, uint _index, string calldata _report) external onlyOwner {
        SubmittedContract storage submitted = userContracts[_user][_index];
        submitted.riskReport = _report;
        submitted.insuranceOffered = true;
        submitted.status = ContractStatus.Rated;
        emit RiskReportUpdated(_user, _index, _report);
    }

    // User accepts insurance
    function acceptInsurance(uint _index) external {
        SubmittedContract storage submitted = userContracts[msg.sender][_index];
        require(submitted.insuranceOffered, "Insurance not offered yet");
        submitted.insuranceAccepted = true;
        submitted.status = ContractStatus.Insured;
        emit InsuranceAccepted(msg.sender, _index);
    }

    // User enables contract monitoring
    function enableMonitoring(uint _index) external {
        SubmittedContract storage submitted = userContracts[msg.sender][_index];
        submitted.monitoringEnabled = true;
        emit MonitoringEnabled(msg.sender, _index);
    }

    // View user submissions
    function getMyContracts() external view returns (SubmittedContract[] memory) {
        return userContracts[msg.sender];
    }
}
