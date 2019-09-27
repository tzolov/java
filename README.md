# TensorFlow for Java

***!!! IMPORTANT NOTICE !!! This repository is UNDER CONSTRUCTION and does not yet host the code of the 
offical TensorFlow Java artifacts!***

***Please refer to the [TensorFlow Java module](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/java) 
of the main repository for the actual code.***

## Welcome to the Java world of TensorFlow!

TensorFlow can run on any JVM for building, training and running machine learning models. It comes with 
a series of utilities and frameworks that help achieve most of the tasks common to data scientists 
and developers working in this domain. Java and other JVM languages, such as Scala or Kotlin, are 
frequently used in small-to-large enterprises all over the world, which makes TensorFlow a strategic 
choice for adopting machine learning at a large scale.

## This Repository

In the early days, the Java language bindings for TensorFlow were hosted in the [main repository](https://github.com/tensorflow/tensorflow)
and released only when a new version of the core library was ready to be distributed, which happens only
a few times a year. Now, all Java-related code has been moved to this repository so that it can evolve and 
be released independently from official TensorFlow releases. In addition, most of the build tasks have been
migrated from Bazel to Maven, which is more familiar for most Java developers.

The following describes the layout of the repository and its different artifacts:

* `core`
  * All artifacts that build up the core language bindings of TensorFlow for Java. 
  * Those artifacts provide the minimal support required to use the TensorFlow runtime on a JVM.
  
* `utils`
  * Utility libraries that do not depend on the TensorFlow runtime but are useful for machine learning purposes
  
* `frameworks`
  * High-level APIs built on top of the core libraries for simplifying the usage of TensorFlow in Java.
  
* `starters`
  * Artifacts aggregating others for simplifying dependency management with TensorFlow
  
*Note: Right now, only the `core` component is present*
  
## Getting Started

Create a new maven Java project and add the following `tensorflow` dependency:

```xml
<dependency>
  <groupId>org.tensorflow</groupId>
  <artifactId>tensorflow</artifactId>
  <version>2.0.0-SNAPSHOT</version>
</dependency>
```

With this you are ready to start using the `Tensorflow Java API` for building new or inferring a pre-built Tensorflow models. 

Following snippets shows how to use the API to build a image recognition service that uses an [Inception model](https://github.com/tensorflow/models/tree/master/inception) to classify in real-time images into different categories (e.g. labels).

The input of the service are the model path and the image location. Result contains the name of the recognized category (e.g. label) along with the confidence (e.g. confidence) that the image represents this category. 

```java
/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.model.sample.eager;

import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;

import org.tensorflow.EagerSession;
import org.tensorflow.Graph;
import org.tensorflow.Operand;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;
import org.tensorflow.op.Ops;
import org.tensorflow.op.image.DecodeJpeg;

/** Sample use of the TensorFlow Java API to label images using a pre-trained model. */
public class LabelImage {

  /** Usage: label_image <model dir> <image file> */
  public static void main(String[] args) {
    String modelDir = args[0];
    String imageFile = args[1];

    byte[] graphDef = readAllBytesOrExit(Paths.get(modelDir, "tensorflow_inception_graph.pb"));
    List<String> labels = readAllLinesOrExit(Paths.get(modelDir, "imagenet_comp_graph_label_strings.txt"));
    byte[] imageBytes = readAllBytesOrExit(Paths.get(imageFile));

    Tensor<Float> image = normalizeImage(imageBytes);
    float[] labelProbabilities = executeInceptionGraph(graphDef, image);
    int bestLabelIdx = maxIndex(labelProbabilities);

    System.out.println(String.format("BEST MATCH: %s (%.2f%% likely)", labels.get(bestLabelIdx), labelProbabilities[bestLabelIdx] * 100f));
  }

  /** Normalize image eagerly */
  private static Tensor<Float> normalizeImage(byte[] imageBytes) {
    try (EagerSession session = EagerSession.create()) {
      Ops tf = Ops.create(session);
      final int H = 224;
      final int W = 224;
      final float mean = 117f;
      final float scale = 1f;      
      final Operand<Float> decodedImage = tf.dtypes.cast(tf.image.decodeJpeg(tf.constant(imageBytes), DecodeJpeg.channels(3L)), Float.class);
      final Operand<Float> resizedImage = tf.image.resizeBilinear(tf.expandDims(decodedImage, tf.constant(0)), tf.constant(new int[] {H, W}));     
      final Operand<Float> normalizedImage = tf.math.div(tf.math.sub(resizedImage, tf.constant(mean)), tf.constant(scale));      
      return normalizedImage.asOutput().tensor();
    }
  }

  private static float[] executeInceptionGraph(byte[] graphDef, Tensor<Float> image) {
    try (Graph g = new Graph()) {
      g.importGraphDef(graphDef);
      try (Session s = new Session(g);
          // Generally, there may be multiple output tensors, all of them must be closed to prevent resource leaks.
          Tensor<Float> result = s.runner().feed("input", image).fetch("output").run().get(0).expect(Float.class)) {
        final long[] rshape = result.shape();
        if (result.numDimensions() != 2 || rshape[0] != 1) {
          throw new RuntimeException(String.format("Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape %s", Arrays.toString(rshape)));
        }
        int nlabels = (int) rshape[1];
        return result.copyTo(new float[1][nlabels])[0];
      }
    }
  }

  private static int maxIndex(float[] probabilities) {
    int best = 0;
    for (int i = 1; i < probabilities.length; ++i) {
      if (probabilities[i] > probabilities[best]) {
        best = i;
      }
    }
    return best;
  }

  private static byte[] readAllBytesOrExit(Path path) {
    try {
      return Files.readAllBytes(path);
    } catch (IOException e) {
      System.err.println("Failed to read [" + path + "]: " + e.getMessage());
      System.exit(1);
    }
    return null;
  }

  private static List<String> readAllLinesOrExit(Path path) {
    try {
      return Files.readAllLines(path, Charset.forName("UTF-8"));
    } catch (IOException e) {
      System.err.println("Failed to read [" + path + "]: " + e.getMessage());
      System.exit(0);
    }
    return null;
  }
}
```

## Details about using Tensorflow Maven artifacts

To include TensorFlow in your Maven application, you first need to add a dependency on both
`tensorflow-core` and `tensorflow-core-native` artifacts. The later could be included multiple times
for different targeted systems by their classifiers.

For example, for building a JAR that uses TensorFlow and is targeted to be deployed only on Linux
systems, you should add the following dependencies:
```xml
<dependency>
  <groupId>org.tensorflow</groupId>
  <artifactId>tensorflow-core</artifactId>
  <version>2.0.0-SNAPSHOT</version>
</dependency>
<dependency>
  <groupId>org.tensorflow</groupId>
  <artifactId>tensorflow-core-native</artifactId>
  <version>2.0.0-SNAPSHOT</version>
  <classifier>linux-x86_64</classifier>
</dependency>
```

On the other hand, if you plan to deploy your JAR on more platforms, you need additional
native dependencies as follows:
```xml
<dependency>
  <groupId>org.tensorflow</groupId>
  <artifactId>tensorflow-core</artifactId>
  <version>2.0.0-SNAPSHOT</version>
</dependency>
<dependency>
  <groupId>org.tensorflow</groupId>
  <artifactId>tensorflow-core-native</artifactId>
  <version>2.0.0-SNAPSHOT</version>
  <classifier>linux-x86_64</classifier>
</dependency>
<dependency>
  <groupId>org.tensorflow</groupId>
  <artifactId>tensorflow-core-native</artifactId>
  <version>2.0.0-SNAPSHOT</version>
  <classifier>windows-x86_64</classifier>
</dependency>
<dependency>
  <groupId>org.tensorflow</groupId>
  <artifactId>tensorflow-core-native</artifactId>
  <version>2.0.0-SNAPSHOT</version>
  <classifier>darwin-x86_64</classifier>
</dependency>
```

In some cases, pre-configured starter artifacts can help to automatically include all versions of
the native library for a given configuration. For example, the `tensorflow` artifact includes
transitively all the artifacts above as a single dependency:
```xml
<dependency>
  <groupId>org.tensorflow</groupId>
  <artifactId>tensorflow</artifactId>
  <version>2.0.0-SNAPSHOT</version>
</dependency>
```

Be aware though that the native library is quite large and including too many versions of it may
significantly increase  the size of your JAR. So it is good practice to limit your dependencies to
the platforms you are targeting.

*Note: the `tensorflow` starter artifact is not available at this moment*

## Building Sources

To build all the artifacts, simply invoke the command `mvn install` at the root of this repository (or 
the Maven command of your choice).

Note that in some cases, if a version of the TensorFlow runtime library is not found for your environment,
this process will fetch TensorFlow sources and trigger a build of all the native code (which can take
many hours on a standard laptop). In this case, you will also need to have a valid environment for building
TensorFlow, including the [bazel](https://bazel.build/) build tool and a few python dependencies. Please
read [TensorFlow documentation](https://www.tensorflow.org/install) for more details.

## How to Contribute?

This repository is maintained by TensorFlow JVM Special Interest Group (SIG). You can easily join the group
by subscribing to the [jvm@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/jvm)
mailing list, or you can simply send pull requests and raise issues to this repository.
