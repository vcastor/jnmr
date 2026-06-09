#!/bin/perl
use strict;
use warnings;

# Urea mass :: 60.0324 Da
# Choline Chloride mass :: 139.0764 Da
# 74 Choline chloride + 151 Urea
my $MASS_DA = (74*139.0764 + 151*60.0324);

# constants
my $DA_TO_G   = 1.66053906660e-24;
my $A3_TO_CM3 = 1.0e-24;

my $file = shift or die "usage: $0 file\n";
open my $fh, '<', $file or die "cannot open $file\n";

my @densities;
my @pressures;
my @temperatures;
my $line = 0;

my $mass_g = $MASS_DA * $DA_TO_G;

while (<$fh>) {
   $line++;
   next if $line <= 7;
   chomp;
   my @f = split;
   next unless defined $f[-1];

   my $vol_a3 = $f[-1];
   next if $vol_a3 == 0.0;
   my $press = $f[-2];
   next if $press == 0.0;
   my $temp = $f[-4];
   next if $temp == 0.0;

   my $density = $mass_g / ($vol_a3 * $A3_TO_CM3);
   push @densities,    $density;
   push @pressures,    $press;
   push @temperatures, $temp;
}

close $fh;

my $n = scalar @densities;
die "no data\n" if $n == 0;

sub mean_std {
   my @v = @_;
   my $n = scalar @v;
   my $sum = 0.0;
   $sum += $_ for @v;
   my $mean = $sum / $n;
   my $var  = 0.0;
   $var += ($_ - $mean)**2 for @v;
   return ($mean, sqrt($var / $n));
}

my ($mean_d, $std_d) = mean_std(@densities);
my ($mean_p, $std_p) = mean_std(@pressures);
my ($mean_t, $std_t) = mean_std(@temperatures);

print "density     (g/cm^3) : $mean_d +/- $std_d\n";
print "pressure    (MPa)    : $mean_p +/- $std_p\n";
print "temperature (K)      : $mean_t +/- $std_t\n";

