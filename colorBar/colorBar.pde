color from =   color(255, 0, 0);
color to = color(0, 0, 255);

void setup(){
  size(400, 400);
  noStroke();
  noLoop();
  background(51);
  colorMode(HSB, 255);
}

void draw(){
  lerpBar(20, 20, 360, 40, from, to);
}

void lerpBar(int x, int y, int w, int h, color... colors) {
  pushStyle();
  for (int i=0; i<w; i++) {
    stroke(lerpColors(float(i)/w, colors));
    line(x+i, y, x+i, y+h);
  }
  popStyle();
}

color lerpColors(float amt, color... colors) {
  if(colors.length==1){
    return colors[0];
  }
  float cunit = 1.0/(colors.length-1);
  return lerpColor(colors[floor(amt / cunit)], colors[ceil(amt / cunit)], amt%cunit/cunit);
}
