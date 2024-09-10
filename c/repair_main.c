#include<stdio.h>
#include<stdlib.h>
int main()
{
  system("cat fff.sh | parallel -j 4");
  //system("( ./../ProB/probcli -model_check tmpfile/6_robotcleaner.prob > nothing6 )& \n ( ./../ProB/probcli -model_check tmpfile/7_robotcleaner.prob > nothing7 )& ");
  printf("Done");
}
