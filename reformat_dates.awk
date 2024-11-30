BEGIN   {OFS="\t"
         print "year", "month", "day"}
        {print $2, $3, $4
         samples[$2] += 1}
END     {OFS=""
         for (year in samples) printf "\n%d sample(s) from %d", samples[year], year
         print ""}
