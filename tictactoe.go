package main

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"path/filepath"
	"reflect"

	"github.com/gin-gonic/gin"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

var emptyBoard = []float32{0, 0, 0, 0, 0, 0, 0, 0, 0}

// GameIA handles TicTacToe machine learning code
type GameIA struct {
	sess  *tf.Session
	graph *tf.Graph
}

// Close removes the TensorFlow session
func (g *GameIA) Close() error {
	return g.sess.Close()
}

// get best movement of prediction
func (g *GameIA) bestMove(initial []float32, predict []float32) (axis []int) {
	// variables
	axis = []int{-1, -1}
	f := float32(0.0)

	// iterate prediction
	for idx, value := range predict {
		// check availability
		if initial[idx] == 0 && value > f {
			// update axis
			axis[0] = idx / 3
			axis[1] = idx % 3

			// set max value
			f = value
		}
	}

	return
}

// Suggest returns a move suggestion for a TicTacToe game
func (g *GameIA) Suggest(board []float32) ([]int, error) {
	// check shape
	if len(board) != 9 {
		return nil, fmt.Errorf("invalid input shape: %d", len(board))
	}

	// first movement
	if reflect.DeepEqual(board, emptyBoard) {
		return []int{2, 0}, nil
	}

	// reshape tensor (1,-1)
	i, err := tf.NewTensor([][]float32{board})
	if err != nil {
		return nil, err
	}

	// run neural network
	output, err := g.sess.Run(
		map[tf.Output]*tf.Tensor{
			g.graph.Operation("main_input_input").Output(0): i,
		},
		[]tf.Output{
			g.graph.Operation("main_output/Sigmoid").Output(0),
		},
		nil)
	if err != nil {
		return nil, err
	}

	// get best movement
	predict := output[0].Value().([][]float32)[0]
	return g.bestMove(board, predict), nil
}

// NewGameIA builds a TensorFlow model from saved data
func NewGameIA() (*GameIA, error) {
	// load a frozen graph to use for queries
	fn := filepath.Join("models", "tictactoe.pb")
	model, err := ioutil.ReadFile(fn)
	if err != nil {
		return nil, err
	}

	// construct an in-memory graph from the serialized form.
	graph := tf.NewGraph()
	if err := graph.Import(model, ""); err != nil {
		return nil, err
	}

	// create a session for inference over graph.
	session, err := tf.NewSession(graph, nil)
	if err != nil {
		return nil, err
	}

	return &GameIA{sess: session, graph: graph}, nil
}

func main() {
	// load AI model
	game, err := NewGameIA()
	if err != nil {
		log.Fatal("cannot load tictactoe model")
	}
	defer game.Close()

	// http server
	gin.SetMode(gin.ReleaseMode)
	r := gin.Default()
	r.GET("/suggest/:data", func(c *gin.Context) {
		// decode base64 string
		enc, err := base64.StdEncoding.DecodeString(c.Param("data"))
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{
				"error": "cannot decode base64 string",
			})
			return
		}

		// decode json
		var board []float32
		if err := json.Unmarshal(enc, &board); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{
				"error": "cannot decode json",
			})
			return
		}

		// machine learning
		res, err := game.Suggest(board)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		// encode json & response
		b, err := json.Marshal(res)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		c.String(http.StatusOK, string(b))
	})

	// frontend
	r.Static("/static", "./static")
	r.StaticFile("/", "./static/index.html")
	r.Run(":5000")
}
