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

   my $density = $mass_g / ($vol_a3 * $A3_TO_CM3);
   push @densities, $density;
}

close $fh;

my $n = scalar @densities;
die "no data\n" if $n == 0;

my $sum = 0.0;
$sum += $_ for @densities;
my $mean_d = $sum / $n;

my $var = 0.0;
$var += ($_ - $mean_d)**2 for @densities;
my $std_d = sqrt($var / $n);

print "Mean density (g/cm^3): $mean_d\n";
print "Std  density (g/cm^3): $std_d\n";

