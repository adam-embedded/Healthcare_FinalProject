package main

import (
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"regexp"
	"strings"
	"time"

	mqtt "github.com/eclipse/paho.mqtt.golang"
)

var serial string = "000006" // This will be read from fuse in production

const HOST = "*****.amazonaws.com"
const PORT = "8883"
const PREFIX = "HA_"

const CERTLOC = "client.crt"
const KEYLOC = "client.key"
const ROOTCA = "AmazonRootCA1.pem"

type IOTClient struct {
	serial         string
	thing          string
	cert           string
	key            string
	connected      bool
	reconnect_time time.Time
	reconnect_n    int8
	onRPC          map[string]func(*IOTClient, string, map[string]interface{}) string
	opts           *mqtt.ClientOptions
	//tlsconfig      *tls.Config
	token mqtt.Token
	cli   mqtt.Client
}

//constructor
func NewIOTClient() *IOTClient {
	c := &IOTClient{}
	c.serial = serial
	c.thing = PREFIX + serial
	c.cert = CERTLOC
	c.key = KEYLOC

	c.onRPC = make(map[string]func(*IOTClient, string, map[string]interface{}) string)

	c.opts = mqtt.NewClientOptions()
	c.opts.AddBroker(fmt.Sprintf("tls://%s:%s", HOST, PORT))
	c.opts.SetClientID(PREFIX + serial)
	c.opts.SetDefaultPublishHandler(c.onMessage)
	c.opts.OnConnect = func(client mqtt.Client) {
		log.Println("AWS connected")
		c.reconnect_time = time.Time{}
		c.reconnect_n = 0
		c.connected = true
		c.subscribe("$aws/things/" + c.thing + "/shadow/name/config/update/delta")
		c.subscribe("$aws/things/" + c.thing + "/shadow/name/config/update/rejected")
		c.subscribe("HA/devices/" + c.thing + "/rpc/request/+")

	}
	c.opts.OnConnectionLost = func(client mqtt.Client, err error) {
		log.Println("AWS disconnected: ", err)
		c.connected = false
		//defs.AWSconnect = false
		//defs.AWSCounter = 0
		c.reconnect_time = time.Now()
	}

	//Initialise connection
	c.connect()
	return c
}

// Helper funcs, should probably improve error handling

func (c *IOTClient) publish(endpoint string, values map[string]interface{}) bool {
	payload, err := json.Marshal(values)
	if err != nil {
		log.Println("Failed to marshal values: ", err)
		return false
	}
	if token := c.cli.Publish(endpoint, 0, false, payload); token.Wait() && token.Error() != nil {
		log.Println("Failed to publish: ", token.Error())
		return false
	}
	return true
}

func (c *IOTClient) subscribe(endpoint string) bool {
	if token := c.cli.Subscribe(endpoint, 0, nil); token.Wait() && token.Error() != nil {
		log.Println("Failed to subscribe: ", token.Error())
		return false
	}
	return true
}

// Paho MQTT callbacks

func (c *IOTClient) onMessage(client mqtt.Client, msg mqtt.Message) {
	fmt.Printf("TOPIC: %s\n", msg.Topic())
	fmt.Printf("MSG: %s\n", msg.Payload())

	if strings.Contains(msg.Topic(), "HA/devices/"+c.thing+"/rpc/request/") {
		parts := strings.Split(msg.Topic(), "/")
		reqNo := parts[len(parts)-1]

		// Check reqNo is alphanumeric
		if !regexp.MustCompile(`^[A-Za-z0-9]+$`).MatchString(reqNo) {
			c.publish("HA/devices/"+c.thing+"/rpc/response/"+reqNo, map[string]interface{}{"success": false, "reason": "Non alpha-numeric request number"})
			return
		}

		// Check RPC
		var rpc struct {
			Cmd    string                 `json:"cmd"`
			Params map[string]interface{} `json:"params"`
		}
		if err := json.Unmarshal(msg.Payload(), &rpc); err != nil {
			c.publish("HA/devices/"+c.thing+"/rpc/response/"+reqNo, map[string]interface{}{"success": false, "reason": "Invalid JSON"})
			return
		}

		log.Println(rpc)

		// Call handler
		handler, handlerOk := c.onRPC[rpc.Cmd]
		reason := "Unknown RPC"
		if handlerOk {
			reason = handler(c, reqNo, rpc.Params)
		}
		if reason == "" {
			c.publish("HA/devices/"+c.thing+"/rpc/response/"+reqNo, map[string]interface{}{"success": true})
		} else {
			c.publish("HA/devices/"+c.thing+"/rpc/response/"+reqNo, map[string]interface{}{"success": false, "reason": reason})
		}
	}

}

// Main funcs

func (c *IOTClient) connect() {
	c.opts.SetCleanSession(true)
	c.reconnect_time = time.Now()

	certstr := CERTLOC
	keystr := KEYLOC

	certpool := x509.NewCertPool()
	pemCerts, err := ioutil.ReadFile(ROOTCA)
	if err != nil {
		log.Println(err)
		return
	}
	certpool.AppendCertsFromPEM(pemCerts)

	cert, err := tls.LoadX509KeyPair(certstr, keystr)
	if err != nil {
		log.Println(err)
		return
	}

	cert.Leaf, err = x509.ParseCertificate(cert.Certificate[0])
	if err != nil {
		log.Println(err)
		return
	}

	c.opts.SetTLSConfig(&tls.Config{
		RootCAs:            certpool,
		ClientAuth:         tls.NoClientCert,
		ClientCAs:          nil,
		InsecureSkipVerify: true,
		Certificates:       []tls.Certificate{cert},
	})

	// Start the connection.
	c.cli = mqtt.NewClient(c.opts)
	if c.token = c.cli.Connect(); c.token.Wait() && c.token.Error() != nil {
		log.Printf("Failed to create connection: %v\n", c.token.Error())
	}
}

func (c *IOTClient) sendFall(values map[string]interface{}) bool {
	return c.publish("HA/devices/"+c.thing+"/fall", values)
}
