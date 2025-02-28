// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract MachineData {
    struct Machine {
        uint8 temperature;
        uint16 speed;
        uint16 torque;
        uint16 wear;
        bool failure;
        bytes32 productID; // More gas-efficient than `string`
        bytes32 mType; // More gas-efficient than `string`
    }

    mapping(uint => Machine) public machines;
    uint public machineCount; // Tracks the total number of machines

    event MachineAdded(
        uint indexed id,
        bytes32 productID,
        bytes32 mType,
        uint8 temperature,
        uint16 speed,
        uint16 torque,
        uint16 wear,
        bool failure
    );

    function addMachine(
        uint _id,
        bytes32 _productID,
        bytes32 _mType,
        uint8 _temperature,
        uint16 _speed,
        uint16 _torque,
        uint16 _wear,
        bool _failure
    ) public {
        require(machines[_id].speed == 0, "Machine with this ID already exists");

        machines[_id] = Machine(_temperature, _speed, _torque, _wear, _failure, _productID, _mType);
        machineCount++;

        emit MachineAdded(_id, _productID, _mType, _temperature, _speed, _torque, _wear, _failure);
    }

    function getMachine(uint _id) public view returns (Machine memory) {
        require(machines[_id].speed != 0, "Machine not found");
        return machines[_id];
    }
}
