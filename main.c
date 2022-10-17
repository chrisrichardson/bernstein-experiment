#include <stdio.h>
#include<stdlib.h>

#include "b.c"


int main()
{
  double f0[nq][nq] = {0};
  for (int i = 0; i < nq; ++i)
    for (int j = 0; j < nq; ++j)
      f0[i][j] = 1.0;

  double A[np][np] = {0};

  tabulate_bernstein_mass_tri(&f0[0][0], &A[0][0]);
  for (int i = 0; i < np; ++i)
  {
    printf("[ ");
    for (int j = 0; j < np; ++j)
      printf("%7.2g ", A[i][j]);
    printf("]\n");
  }
}
