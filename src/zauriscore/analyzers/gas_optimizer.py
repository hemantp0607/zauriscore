from typing import List, Dict, Any, Union
from slither import Slither
from slither.core.declarations import Contract, Structure
from slither.core.variables.state_variable import StateVariable
from slither.slithir.variables import Constant
from slither.core.cfg.node import Node
from slither.core.expressions.expression import Expression
import logging
import tempfile
import os

class GasOptimizationAnalyzer:
    def __init__(self):
        self.optimizations = []

    def analyze_storage_packing(self, contract: Contract) -> List[Dict[str, Any]]:
        """Analyze storage packing opportunities."""
        optimizations = []
        
        # Get consecutive state variables that could be packed
        consecutive_vars = []
        current_group = []
        
        for var in contract.state_variables:
            if var.visibility in ['public', 'internal', 'private']:
                var_size = self._get_variable_size(var.type)
                
                # Start new group if current variable is too large or group would exceed 32 bytes
                if var_size > 32 or (current_group and sum(self._get_variable_size(v.type) for v in current_group) + var_size > 32):
                    if len(current_group) > 1:
                        consecutive_vars.append(current_group)
                    current_group = [var]
                else:
                    current_group.append(var)
        
        # Don't forget the last group
        if len(current_group) > 1:
            consecutive_vars.append(current_group)
        
        # Suggest optimizations for groups that could be better packed
        for group in consecutive_vars:
            total_size = sum(self._get_variable_size(v.type) for v in group)
            if total_size <= 32 and len(group) > 1:
                optimizations.append({
                    'issue': 'Storage Packing Opportunity',
                    'severity': 'medium',
                    'suggestion': 'These variables can be packed into a single storage slot',
                    'saving': '~2000-5000 gas per slot saved',
                    'category': 'storage',
                    'matched_code': ', '.join([v.name for v in group]),
                    'example_before': '\n'.join([f'{v.type} {v.name};' for v in group]),
                    'example_after': '// These variables are packed together:\n' +
                                    '\n'.join([f'{v.type} {v.name};' for v in group]),
                    'rationale': 'Multiple small variables can be packed into a single storage slot to save gas'
                })

        return optimizations

    def analyze(self, contract: Union[Contract, str]) -> List[Dict[str, Any]]:
        """Analyze a contract for gas optimization opportunities.
        
        Args:
            contract: Either a Slither Contract object or a string containing Solidity source code
            
        Returns:
            List of optimization opportunities
        """
        optimizations = []
        
        # If input is a string, parse it as Solidity source code
        if isinstance(contract, str):
            try:
                # First try direct string analysis
                string_optimizations = self.analyze_solidity_source(contract)
                
                # Try Slither analysis for more detailed results
                with tempfile.NamedTemporaryFile(mode='w', suffix='.sol', delete=False) as temp:
                    temp.write(contract)
                    temp_path = temp.name
                
                try:
                    slither = Slither(temp_path)
                    if slither.contracts:
                        for contract_obj in slither.contracts:
                            slither_optimizations = self.analyze_contract(contract_obj)
                            # Merge with string analysis, avoiding duplicates
                            optimizations.extend(slither_optimizations)
                    
                    # If we got results from Slither, use those; otherwise use string analysis
                    return optimizations if optimizations else string_optimizations
                    
                except Exception as slither_error:
                    logging.info(f"Slither analysis failed, using string analysis: {slither_error}")
                    return string_optimizations
                
            except Exception as e:
                logging.error(f"Error in contract analysis: {e}")
                return []
                
            finally:
                # Ensure cleanup happens
                if 'temp_path' in locals():
                    try:
                        os.unlink(temp_path)
                    except Exception:
                        pass  # Ignore cleanup errors
        
        # If input is already a Contract object
        elif isinstance(contract, Contract):
            return self.analyze_contract(contract)
            
        else:
            raise ValueError("Input must be either a Solidity source code string or a Slither Contract object")
    
    def analyze_contract(self, contract: Contract) -> List[Dict[str, Any]]:
        """Analyze a single contract for all gas optimizations."""
        optimizations = []
        optimizations.extend(self.analyze_storage_packing(contract))
        optimizations.extend(self.analyze_public_mappings(contract))
        optimizations.extend(self.analyze_struct_packing(contract))
        optimizations.extend(self.analyze_dynamic_bytes(contract))
        optimizations.extend(self.analyze_mapping_initialization(contract))
        optimizations.extend(self.analyze_multiple_small_uints(contract))
        return optimizations
    
    def analyze_multiple_small_uints(self, contract: Contract) -> List[Dict[str, Any]]:
        """Detect multiple small uint variables that could be packed together."""
        optimizations = []
        
        # Group variables by their types
        uint_vars = []
        for var in contract.state_variables:
            if var.visibility in ['public', 'internal', 'private']:
                var_type_str = str(var.type)
                if var_type_str.startswith('uint') and var_type_str != 'uint256':
                    uint_vars.append(var)
        
        # If we have multiple small uints, suggest packing
        if len(uint_vars) > 1:
            optimizations.append({
                'issue': 'Multiple Small Uints',
                'severity': 'medium',
                'suggestion': 'Group smaller uints together to save storage slots',
                'saving': '~2000-5000 gas per slot saved',
                'category': 'storage',
                'matched_code': ', '.join([f'{var.type} {var.name}' for var in uint_vars]),
                'example_before': '\n'.join([f'{var.type} {var.name};' for var in uint_vars]),
                'example_after': '// Consider grouping these together:\n' + '\n'.join([f'{var.type} {var.name};' for var in uint_vars]),
                'rationale': 'Grouping smaller uints together can reduce the number of storage slots used.'
            })
        
        return optimizations
    
    def analyze_solidity_source(self, source_code: str) -> List[Dict[str, Any]]:
        """Analyze Solidity source code using string matching when Slither parsing fails."""
        optimizations = []
        
        # Check for public mappings
        if 'mapping(' in source_code and 'public' in source_code and ';' in source_code.split('public')[-1]:
            optimizations.append({
                'issue': 'Public Mapping',
                'severity': 'low',
                'suggestion': 'Consider making the mapping private and adding a getter function',
                'saving': '~2000 gas per access',
                'category': 'storage',
                'matched_code': 'mapping(...) public ...',
                'example_before': 'mapping(...) public name;',
                'example_after': 'mapping(...) private _name;\n\n    function getName(...) public view returns (...) {\n        return _name[...];\n    }',
                'rationale': 'Public mappings generate an implicit getter function. Using a private mapping with an explicit getter can save gas.'
            })
        
        # Check for struct packing opportunities
        if 'struct' in source_code and '}' in source_code.split('struct')[-1]:
            optimizations.append({
                'issue': 'Struct Packing Opportunity',
                'severity': 'medium',
                'suggestion': 'Reorganize struct fields to use fewer storage slots',
                'saving': '~2000-5000 gas per slot saved',
                'category': 'storage',
                'matched_code': 'struct ... { ... }',
                'example_before': 'struct Example {\n    bool active;\n    uint256 id;\n    address user;\n    uint8 age;\n}',
                'example_after': 'struct Example {\n    uint256 id;\n    address user;\n    bool active;\n    uint8 age;\n}',
                'rationale': 'Reordering struct fields can reduce the number of storage slots used, saving gas.'
            })
        
        # Check for multiple small uints that could be packed
        if ('uint8' in source_code or 'uint16' in source_code or 'uint32' in source_code) and \
           ('uint256' in source_code or 'uint128' in source_code):
            optimizations.append({
                'issue': 'Multiple Small Uints',
                'severity': 'medium',
                'suggestion': 'Group smaller uints together to save storage slots',
                'saving': '~2000-5000 gas per slot saved',
                'category': 'storage',
                'matched_code': 'uint8/uint16/uint32 variables',
                'example_before': 'uint8 a;\nuint256 b;\nuint8 c;',
                'example_after': 'uint8 a;\nuint8 c;\nuint256 b;',
                'rationale': 'Grouping smaller uints together can reduce the number of storage slots used.'
            })
        
        # Check for dynamic bytes arrays
        if 'bytes ' in source_code and '=' in source_code.split('bytes ')[-1] and 'new bytes' in source_code:
            optimizations.append({
                'issue': 'Dynamic Bytes Array',
                'severity': 'medium',
                'suggestion': 'Use fixed-size bytes (bytes1 to bytes32) instead of dynamic bytes if the maximum size is known',
                'saving': '~20000 gas for storage, ~100 gas per access',
                'category': 'storage',
                'matched_code': 'bytes public name = new bytes(...);',
                'example_before': 'bytes public data = new bytes(20);',
                'example_after': 'bytes20 public data;  // If max size is 20 bytes',
                'rationale': 'Fixed-size bytes are more gas-efficient than dynamic bytes arrays.'
            })
        
        # Check for mapping initialization
        if 'mapping(' in source_code and '=' in source_code.split('mapping(')[-1] and '{' in source_code:
            optimizations.append({
                'issue': 'Mapping with Initial Value',
                'severity': 'low',
                'suggestion': 'Initialize mappings in the constructor instead of at declaration',
                'saving': '~20000 gas per mapping',
                'category': 'deployment',
                'matched_code': 'mapping(...) public name = { ... };',
                'example_before': 'mapping(address => bool) public whitelist = {\n    0x123...: true\n};',
                'example_after': 'mapping(address => bool) public whitelist;\n\n    constructor() {\n        whitelist[0x123...] = true;\n    }',
                'rationale': 'Initializing mappings in the constructor is more gas-efficient than at declaration.'
            })
        
        return optimizations

    def analyze_public_mappings(self, contract: Contract) -> List[Dict[str, Any]]:
        """Detect public mappings that could be made private."""
        optimizations = []
        
        for var in contract.state_variables:
            if var.visibility == 'public' and str(var.type).startswith('mapping'):
                # Check if there's a getter function that could replace the public mapping
                has_getter = any(
                    f.is_constructor is False and 
                    f.visibility in ['public', 'external'] and 
                    f.pure or f.view
                    for f in contract.functions
                )
                
                if not has_getter:
                    optimizations.append({
                        'issue': 'Public Mapping',
                        'severity': 'low',
                        'suggestion': f'Consider making mapping {var.name} private and adding a getter function',
                        'saving': '~2000 gas per access',
                        'category': 'storage',
                        'matched_code': f'mapping(...) public {var.name}' if hasattr(var, 'name') else str(var),
                        'example_before': f'mapping(...) public {var.name};',
                        'example_after': f'mapping(...) private {var.name};\n\n    function get{var.name.capitalize()}(...) public view returns (...) {{\n        return {var.name}[...];\n    }}',
                        'rationale': 'Public mappings generate an implicit getter function. Using a private mapping with an explicit getter can save gas.'
                    })
        
        return optimizations

    def analyze_struct_packing(self, contract: Contract) -> List[Dict[str, Any]]:
        """Detect inefficient struct packing."""
        optimizations = []
        
        for struct in contract.structures:
            # Get all variables in the struct
            vars_in_struct = struct.elems_ordered
            
            # Calculate current storage usage
            current_slots = 0
            current_slot_used = 0
            
            for var in vars_in_struct:
                var_size = self._get_variable_size(var.type)
                
                if current_slot_used + var_size > 32:
                    current_slots += 1
                    current_slot_used = var_size
                else:
                    current_slot_used += var_size
            
            if current_slot_used > 0:
                current_slots += 1
            
            # Calculate optimal packing
            vars_sorted = sorted(vars_in_struct, key=lambda v: self._get_variable_size(v.type), reverse=True)
            optimal_slots = 0
            optimal_used = 0
            
            for var in vars_sorted:
                var_size = self._get_variable_size(var.type)
                
                if optimal_used + var_size > 32:
                    optimal_slots += 1
                    optimal_used = var_size
                else:
                    optimal_used += var_size
            
            if optimal_used > 0:
                optimal_slots += 1
            
            # If we can save slots, add an optimization
            if optimal_slots < current_slots:
                optimizations.append({
                    'issue': 'Inefficient Struct Packing',
                    'severity': 'medium',
                    'suggestion': f'Reorganize struct {struct.name} to use fewer storage slots',
                    'saving': f'~{2000 * (current_slots - optimal_slots)} gas per instance',
                    'category': 'storage',
                    'matched_code': f'struct {struct.name} {{ ... }}',
                    'example_before': f'struct {struct.name} {{\n    ' + '\n    '.join(f'{var.type} {var.name};' for var in vars_in_struct) + '\n}',
                    'example_after': f'struct {struct.name} {{\n    ' + '\n    '.join(f'{var.type} {var.name};' for var in vars_sorted) + '\n}',
                    'rationale': 'Reordering struct variables can reduce storage slots used, saving gas.'
                })
        
        return optimizations

    def analyze_dynamic_bytes(self, contract: Contract) -> List[Dict[str, Any]]:
        """Detect dynamic bytes arrays that could be fixed-size."""
        optimizations = []
        
        for var in contract.state_variables:
            if str(var.type) == 'bytes' and var.visibility in ['public', 'private', 'internal']:
                optimizations.append({
                    'issue': 'Dynamic Bytes Array',
                    'severity': 'medium',
                    'suggestion': f'Consider using bytes32 instead of dynamic bytes for {var.name} if the maximum size is known',
                    'saving': '~20000 gas for storage, ~100 gas per access',
                    'category': 'storage',
                    'matched_code': f'bytes public {var.name};',
                    'example_before': f'bytes public {var.name};',
                    'example_after': f'bytes32 public {var.name};  // If max size is 32 bytes',
                    'rationale': 'Dynamic bytes arrays are more expensive in terms of gas than fixed-size bytes.'
                })
        
        return optimizations

    def analyze_mapping_initialization(self, contract: Contract) -> List[Dict[str, Any]]:
        """Detect mappings that are initialized with values."""
        optimizations = []
        
        for var in contract.state_variables:
            if str(var.type).startswith('mapping') and var.expression:
                optimizations.append({
                    'issue': 'Mapping with Initial Value',
                    'severity': 'low',
                    'suggestion': f'Initialize mapping {var.name} in the constructor instead of at declaration',
                    'saving': '~20000 gas per mapping',
                    'category': 'deployment',
                    'matched_code': f'mapping(...) public {var.name} = ...;',
                    'example_before': f'mapping(...) public {var.name} = {{ ... }};',
                    'example_after': f'mapping(...) public {var.name};\n\n    constructor() {{\n        {var.name}[...] = ...;\n    }}',
                    'rationale': 'Initializing mappings at declaration is more expensive than in the constructor.'
                })
        
        return optimizations

    def _get_variable_size(self, var_type: Union[str, Any]) -> int:
        """Get the size of a variable type in bytes."""
        type_str = str(var_type)
        
        if type_str.startswith('uint'):
            # For uint types, get the size from the type name
            if type_str == 'uint':
                return 32  # uint defaults to uint256
            size_bits = type_str[4:]
            if size_bits.isdigit():
                return int(size_bits) // 8
            return 32
        elif type_str == 'bool':
            return 1
        elif type_str.startswith('bytes'):
            if type_str == 'bytes':
                return 32  # Dynamic bytes, treat as 32 for packing analysis
            size_str = type_str[5:]
            if size_str.isdigit():
                return int(size_str)
            return 32
        elif type_str == 'address':
            return 20
        
        return 32  # Default to 32 bytes for unknown types