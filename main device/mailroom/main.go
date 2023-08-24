package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net"
	"time"
)

//Healthcare robot mailroom
//This program takes messages from all running processes and redirects them to their desired location
// this is a UDP socket server, also sends data to AWS IOT via MQTT.

func main() {
	//initialise the AWS client, this is single threaded so is blocking
	c := NewIOTClient()

	serverAddr, err := net.ResolveUDPAddr("udp", "localhost:12345")
	if err != nil {
		fmt.Println("Error resolving address:", err)
		return
	}
	udpServer, err := net.ListenUDP("udp", serverAddr)
	if err != nil {
		log.Fatal(err)
	}

	defer udpServer.Close()
	fmt.Println("UDP Server is listening on", serverAddr)

	//create a list of client so that we know who to send messages back to.
	clients := make(map[string]*net.UDPAddr)

	//infinite loop
	for {
		buf := make([]byte, 1024)
		n, addr, err := udpServer.ReadFromUDP(buf)
		if n < 0 {
			fmt.Println("There is no data from socket")
		}
		if err != nil {
			continue
		}

		message := string(buf[:n])

		fmt.Printf("Received this message from %s: %s\n", addr, message)

		if message != "" {

			//check that the message is not for the server.
			var control struct {
				Id     int `json:"Id"`
				Mess   int `json:"message"`
				Intent int `json:"intent"`
			}

			err := json.Unmarshal(buf, &control)
			if err != nil {
				fmt.Println("error opening json")
				continue
			}
			//check the intent of the message, 1 is the mailroom
			if control.Intent != 1 {
				//Broadcast to all connected clients other than sender
				for _, add := range clients {
					if add.String() != addr.String() {
						_, err := udpServer.WriteToUDP([]byte(message), addr)
						if err != nil {
							fmt.Errorf("Error Broadcasting: %s", err)
						}
					}
				}
			} else {
				//message intended for mailroom, check what the message is and who it is from
				if control.Mess == 1 && control.Id == 2 {
					//message is to show the user has fallen, send a true message for fallen with a timestamp
					c.sendFall(map[string]interface{}{"ts": time.Now().Unix(), "value": true})
				}
			}
		}

		clients[addr.String()] = addr

	}

}
