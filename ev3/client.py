import socket

def send_commands():
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(("192.168.3.2", 5000))  # Connect to EV3 server

    ready_msg = client.recv(1024).decode().strip()
    if ready_msg == "READY":
        print("Connected to EV3. You can now send commands (e.g., TURN 50 180, RUN 30 3, STOP, EXIT).")

        while True:
            command = input("Enter command: ").strip()
            if command.upper() == "EXIT":
                break

            client.sendall(f"{command}\n".encode())
            response = client.recv(1024).decode().strip()
            print("EV3 Response:", response)

    client.close()

# Start the client
send_commands()
