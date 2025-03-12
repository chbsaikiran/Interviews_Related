gcc -o server server.c
gcc -o client client.c
./server
./client

Expected Output:

Server Terminal:
Server listening on port 8080...
Client message: Hello from client!
Message sent to client.

Client Terminal:
Message sent to server.
Server response: Hello from server!
