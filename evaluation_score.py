import tensorflow as tf

tfgan=tf.contrib.gan

def preprocess_inception(input_image):
# make it 3 channel 
    input_image_3_channel=tf.tile(input_image,[1,1,1,3])
# make it 299 X 299
    input_image_reshape=tf.image.resize_bilinear(input_image_3_channel, [299, 299],
                                           align_corners=False)
# make it between -1 to 1 
    input_image_reshape = tf.subtract(input_image_reshape, 0.5)
    input_image_ = tf.multiply(input_image_reshape, 2.0)
    # pdb.set_trace()
    return input_image_

def frechet_distance(real_image,generated_image,num_batches=1):
    real_image=preprocess_inception(real_image)
    generated_image=preprocess_inception(generated_image)
    input_tensor = 'input:0' #shape=[?,299,299,3]
    output_tensor = 'InceptionV4/Logits/AvgPool_1a/AvgPool:0' #shape=[?,1,1,1536]
    graph_def=tfgan.eval.get_graph_def_from_disk('/pb_files/inception_v4_fid.pb')
    # image_net_classifier_fn = lambda x: tfgan.eval.run_image_classifier(  # pylint: disable=g-long-lambda
    #   x, graph_def, input_tensor, output_tensor)
    def image_net_classifier_fn(x):
      output= tfgan.eval.run_image_classifier(  # pylint: disable=g-long-lambda
       x, graph_def, input_tensor, output_tensor)
    # output_reshaped=tf.squeeze(output)
     # pdb.set_trace()
#      output_reshaped = tf.reshape(tensor=output, shape=(-1, tf.shape(output)[1]*tf.shape(output)[2]*tf.shape(output)[3]))
      output_reshaped = tf.reshape(tensor=output, shape=(-1, output.get_shape()[1]*output.get_shape()[2]*output.get_shape()[3]))
      
      #pdb.set_trace()
      return output_reshaped
    frechet_distance = tfgan.eval.frechet_classifier_distance(
      real_image, generated_image ,image_net_classifier_fn,num_batches)
    frechet_distance.shape.assert_is_compatible_with([])
    return frechet_distance

def inception_score(generated_image,num_batches=1):
    generated_image=preprocess_inception(generated_image)
    input_tensor = 'input:0' #shape=[?,299,299,3]
    output_tensor = 'InceptionV4/Logits/Logits/BiasAdd:0' #shape=[?,1001]
    graph_def=tfgan.eval.get_graph_def_from_disk('/pb_files/inception_v4_is.pb')
    image_net_classifier_fn = lambda x: tfgan.eval.run_image_classifier(x, graph_def, input_tensor, output_tensor)
    inception_score=tf.contrib.gan.eval.classifier_score(
                                                        generated_image,
                                                        image_net_classifier_fn,
                                                       num_batches=num_batches
                                                        )
    inception_score.shape.assert_is_compatible_with([])
    
    return inception_score

