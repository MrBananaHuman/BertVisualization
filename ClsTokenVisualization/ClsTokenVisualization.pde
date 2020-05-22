import java.util.Collections;
import java.net.URLEncoder;
import java.io.UnsupportedEncodingException;

color from =   color(255, 0, 0);
color to = color(0, 0, 255);

PFont myFont;

int fontSize = 50;

String inputSent = "";
String encodedInput = "";
int layer = -1;


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

ArrayList<Float> getVector(String json){
  JSONObject jsonObj = parseJSONObject(json);
  int layer = jsonObj.getInt("return_layer");
  String inputSent = jsonObj.getString("input_sentence");
  JSONArray tokens = jsonObj.getJSONArray("tokens");
  JSONArray vectors = jsonObj.getJSONArray("vectors");
  ArrayList<Float> result = new ArrayList<Float>();
  println(layer);
  println(inputSent);
  println(tokens);
  for(int i = 0; i < vectors.size(); i++){
    float vector_value = float(vectors.get(i).toString());
    result.add(vector_value);
  }
  return result;
}


String getTokens(String json){
  JSONObject jsonObj = parseJSONObject(json);
  JSONArray tokens = jsonObj.getJSONArray("tokens");
  String result = "";
  for(int i = 0; i < tokens.size(); i++){
    result += tokens.get(i).toString() + "    ";
  }
  return result.trim();
}

void lerpBar(int x, int y, int w, int h, color... colors) {
  pushStyle();
  for (int i=0; i<w; i++) {
    stroke(lerpColors(float(i)/w, colors));
    line(x+i, y, x+i, y+h);
  }  
}

color lerpColors(float amt, color... colors) {
  if(colors.length==1){
    return colors[0];
  }
  float cunit = 1.0/(colors.length-1);
  return lerpColor(colors[floor(amt / cunit)], colors[ceil(amt / cunit)], amt%cunit/cunit);
}

void setup() {
  fullScreen();
  noStroke();
  background(51);
  colorMode(HSB, 255);
  myFont = createFont("FreeSans Bold", fontSize);
  inputSent = "Attention is all you need!";
  encodedInput = URLEncode(inputSent);
  String[] data = loadStrings("http://127.0.0.1:9090/visualization?sentence=" + encodedInput + "&layer=" + layer);
  textFont(myFont);
  textAlign(CENTER);
  //text(inputSent, width/2, 50);
  String tokens = getTokens(data[0]);
  text(tokens, width/2, 80);
  lerpBar(width/2 - 180, 100, 360, 20, to, from);
  textSize(10);
  text("0", width/2 - 180, 130);
  text("1", width/2 + 180, 130);
  
  
}

void draw(){
  textSize(30);
  textAlign(RIGHT);
  String layer_num = "";
  if(layer == -1){
    layer_num = "emb";
  } else {
    layer_num = str(layer + 1);
  }
  text(layer_num, 100, 200 + 60*(layer+1));
  String[] data = loadStrings("http://127.0.0.1:9090/visualization?sentence=" + encodedInput + "&layer=" + layer);
  ArrayList<Float> clsVector = getVector(data[0]);
  for(int i = 0; i < clsVector.size(); i++){
    color vectorColor = lerpColors(clsVector.get(i), to, from);
    pushStyle();
    strokeWeight(3);
    stroke(vectorColor);
    line(150 + (i * 2.2), 200 + 60*(layer+1)-30, 150 + (i * 2.2), 200 + 60*(layer+1) + 10);
    popStyle();
  }
  if(layer == 11){
    textSize(15);
    textAlign(LEFT);
    text("bananaband657@gmail.com", width - 300, height - 50);
    noLoop();
  }
  if(layer < 11){
    layer += 1;
  }
}
