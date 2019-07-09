This file stores and generated the frozen graph of imagenet.
Only inception_v4 model is available in networks.
To generate the frozen graph following steps need to be done :
1. Run pbtxt.py this file generates the the graph in pbtxt format for inception_v4 model.
2. Next run the freeze_graph.py wich downloads the inception_v4 checkpoint and takes the above as input along with output nodes and produces the frozen graph in pb format.
3. For Inception Score the output node in inception_v4 model is  InceptionV4/Logits/Logits/BiasAdd and for Frechet Inception Distance the output node is InceptionV4/Logits/AvgPool_1a/AvgPool

python pbtxt.py    --alsologtostderr   --model_name=inception_v4   --output_file=./inception_v4.pbtxt --datasett_dir=./

python freeze_graph.py --input_graph=inception_v4.pbtxt --output_graph=inception_v4_is.pb --output_node_names=InceptionV4/Logits/Logits/BiasAdd 

