import socket
from ev3dev2.motor import LargeMotor, OUTPUT_A, SpeedPercent
from time import sleep

# Initialize motor
motor = LargeMotor(OUTPUT_A)

# Set up the TCP server
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("0.0.0.0", 5000))  # Listening on port 5000
server.listen(5)  # Allows up to 5 clients to wait in queue

print("Server is running... Waiting for connections.")

while True:
    conn, addr = server.accept()
    print("Client connected:", addr)

    try:
        conn.sendall("READY\n".encode())  # Notify client that the server is ready

        while True:
            data = conn.recv(1024).decode().strip()
            if not data:
                break  # Disconnect if no data received

            print("Received command:", data)
            response = "Error: Unknown command"

            if data.startswith("TURN "):  # Rotate motor by degrees at speed
                try:
                    _, speed, degrees = data.split()
                    speed = int(speed)
                    degrees = int(degrees)
                    motor.on_for_degrees(SpeedPercent(speed), degrees)
                    response = "Motor rotated {} degrees at {}% speed".format(degrees, speed)
                except ValueError:
                    response = "Error: Invalid TURN command. Use: TURN <speed> <degrees>"

            elif data.startswith("RUN "):  # Run motor at speed for time
                try:
                    _, speed, duration = data.split()
                    speed = int(speed)
                    duration = float(duration)
                    motor.on(SpeedPercent(speed))
                    sleep(duration)
                    motor.off()
                    response = "Motor ran at {}% speed for {} seconds".format(speed, duration)
                except ValueError:
                    response = "Error: Invalid RUN command. Use: RUN <speed> <seconds>"

            elif data == "STOP":
                motor.off()
                response = "Motor stopped"

            conn.sendall(response.encode())

    except Exception as e:
        print("Error:", e)

    finally:
        print("Client disconnected")
        conn.close()  # Close the connection
