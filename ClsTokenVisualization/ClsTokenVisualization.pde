import java.util.Collections;
import java.net.URLEncoder;
import java.io.UnsupportedEncodingException;


PFont myFont;

int fontSize = 10;

ArrayList<ArrayList<Float>> vector = new ArrayList<ArrayList<Float>>();

String URLEncode(String string) {
  String output = new String();
  try {
    byte[] input = string.getBytes("UTF-8");
    for (int i=0; i<input.length; i++) {
      if (input[i]<0)
        output += '%' + hex(input[i]);
      else if (input[i]==32)
        output += '+';
      else
        output += char(input[i]);
    }
  }
  catch(UnsupportedEncodingException e) {
    e.printStackTrace();
  }
  return output;
}

ArrayList<ArrayList<Float>> getVector(String json){
  JSONObject jsonObj = parseJSONObject(json);
  int layer = jsonObj.getInt("return_layer");
  String inputSent = jsonObj.getString("input_sentence");
  JSONArray tokens = jsonObj.getJSONArray("tokens");
  JSONArray vectors = jsonObj.getJSONArray("vectors");
  println(layer);
  println(inputSent);
  println(tokens);
  for(int i = 0; i < vectors.size(); i++){
    float vector_value = float(vectors.get(i).toString());
    println(vector_value * 10);
  }
  
  return null;
}


void setup() {
  //fullScreen(2);
  background(25, 25, 25);
  String inputSent = "hello world!";
  int layer = 0;

  myFont = createFont("FreeSans Bold", fontSize);
  String encodedInput = URLEncode(inputSent);
  String[] data = loadStrings("http://127.0.0.1:9090/visualization?sentence=" + encodedInput + "&layer=" + layer);
  textFont(myFont);
  //println(data[0]);
  getVector(data[0]);
}
