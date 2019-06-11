#include<Servo.h>
#include<String.h>

long int br[]={100000,101000,110000,110100,100100,111000,111100,101100,011000,011100,100010,101010,110010,110110,100110,111110,101110,011010,011110,100011,101011,011101,110011,110111,100111};
long int brn[]={010111,100000,101000,110000,110100,100100,111000,111100,101100,011000,011100};

//char ch = 0;
int sp1 = 3; //servoPin
int sp2 = 5;
int sp3 = 6;
int sp4 = 9;
int sp5 = 10;
int sp6 = 11;
Servo sn[6]; //servoNumber

void setup() {
  Serial.begin(9600);
  sn[0].attach(sp1);
  sn[1].attach(sp2);
  sn[2].attach(sp3);
  sn[3].attach(sp4);
  sn[4].attach(sp5);
  sn[5].attach(sp6);
  Serial.println("Connection Established...");
}

void loop() {
  for(int i=0;i<6;i++)
    sn[i].write(0);
  
  while(Serial.available()){
    char ch = Serial.read();//receive serially from OCR
  
  
  //char ch = 'e ';//receive serially from OCR
  //delay(5000);
  if(ch>='a'&&ch<='z')
  {
    for(long int i=br[ch-'a'],j=5;i>0,j>=0;i/=10,j--)
    {
      
      if(i%10==1){
        sn[j].write(90);
        Serial.println(i%10);
      }
       else
        sn[j].write(0);
    }
  }  
  delay(1000);
}
}
