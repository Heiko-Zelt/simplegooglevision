package simplegooglevision;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import com.google.cloud.vision.v1.AnnotateImageRequest;
import com.google.cloud.vision.v1.AnnotateImageResponse;
import com.google.cloud.vision.v1.BatchAnnotateImagesResponse;
import com.google.cloud.vision.v1.Feature;
import com.google.cloud.vision.v1.Feature.Type;
import com.google.cloud.vision.v1.Image;
import com.google.cloud.vision.v1.ImageAnnotatorClient;
import com.google.cloud.vision.v1.LocalizedObjectAnnotation;
import com.google.protobuf.ByteString;

public class ObjectDetectorExample {
	
	static final String filePath = "/home/heiko/Pictures/IMG_5148.jpg";
	
	public static void main(String[] args) throws IOException {
	 List<AnnotateImageRequest> requests = new ArrayList<>();

	  ByteString imgBytes = ByteString.readFrom(new FileInputStream(filePath));

	  Image img = Image.newBuilder().setContent(imgBytes).build();
	  AnnotateImageRequest request =
	      AnnotateImageRequest.newBuilder()
	          .addFeatures(Feature.newBuilder().setType(Type.OBJECT_LOCALIZATION))
	          .setImage(img)
	          .build();
	  requests.add(request);

	  // Initialize client that will be used to send requests. This client only needs to be created
	  // once, and can be reused for multiple requests. After completing all of your requests, call
	  // the "close" method on the client to safely clean up any remaining background resources.
	  try (ImageAnnotatorClient client = ImageAnnotatorClient.create()) {
	    // Perform the request
	    BatchAnnotateImagesResponse response = client.batchAnnotateImages(requests);
	    List<AnnotateImageResponse> responses = response.getResponsesList();

	    // Display the results
	    for (AnnotateImageResponse res : responses) {
	      for (LocalizedObjectAnnotation entity : res.getLocalizedObjectAnnotationsList()) {
	        System.out.format("Object name: %s%n", entity.getName());
	        System.out.format("  Confidence: %s%n", entity.getScore());
	        System.out.format("  Normalized Vertices:%n");
	        entity
	            .getBoundingPoly()
	            .getNormalizedVerticesList()
	            .forEach(vertex -> System.out.format("  - (%s, %s)%n", vertex.getX(), vertex.getY()));
	      }
	    }
	  }
	}
}
