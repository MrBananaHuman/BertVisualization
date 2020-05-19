color from =   color(255, 0, 0);
color to = color(0, 0, 255);
float colorNum = 0;
void setup(){
  size(400, 400);
  noStroke();
  background(51);
  colorMode(HSB, 255);
  frameRate(30);
}

void draw(){
  lerpBar(20, 20, 360, 40, to, from);
  colorNum += 0.01;
  if(colorNum > 1){
    colorNum = 0;
  }
}

void lerpBar(int x, int y, int w, int h, color... colors) {
  pushStyle();
  for (int i=0; i<w; i++) {
    stroke(lerpColors(float(i)/w, colors));
    line(x+i, y, x+i, y+h);
  }
  popStyle();
  pushStyle();
  fill(lerpColors(colorNum, colors));
  rect(x, y + h + 30, w, height - 100);
  popStyle();
  
}

color lerpColors(float amt, color... colors) {
  if(colors.length==1){
    return colors[0];
  }
  float cunit = 1.0/(colors.length-1);
  return lerpColor(colors[floor(amt / cunit)], colors[ceil(amt / cunit)], amt%cunit/cunit);
}
