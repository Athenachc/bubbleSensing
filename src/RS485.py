#By bobobobobobobobo
import serial
import time
import struct


def sensor_init(serial_port, reset):
    try:
        ser = serial.Serial(
            port=serial_port,
            baudrate=115200,
            bytesize=8,
            parity='N',
            stopbits=1,
            timeout=1
        )

        if ser.is_open:
            print(f"Serial port {serial_port} opened successfully.")
            print("Press Ctrl+C to stop.\n")
            ser.write(reset)
            print("Reset successfully.\n")
            time.sleep(1)
            return ser
        else:
            print(f"Failed to open serial port {serial_port}.")
            return None

    except serial.SerialException as e:
        print(f"Error: {e}")
        return None
        
def sensor_send(ser, data):
    if ser and ser.is_open:
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        ser.write(data)
        #print(f"Sent: {command}")
    else:
        print("Serial port is not open")

def sensor_read(ser, expected_bytes=9):
    if ser and ser.is_open:
        received_data = ser.read(expected_bytes)
        if len(received_data) == expected_bytes:
            # received_data = ser.read(ser.in_waiting)
            received_data_hex = received_data.hex()
            #print(f"Received: {received_data.hex()}")
            value_hex = received_data_hex[6:-4]
            value_bytes = bytes.fromhex(value_hex)
            value = struct.unpack('>f', value_bytes)[0]
            value = value * 9.81
            if value < 0:
                value = 0
            return round(value, 3)
        else:
            print("No data received.")
            return None
    else:
        print("Serial port is not open.")
        return None

def sensor_close(ser):
    if ser and ser.is_open:
        ser.close()
        print(f"Serial port {ser} closed.")