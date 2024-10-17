# download tetrad and xom
wget --no-check-certificate https://cloud.ccd.pitt.edu/nexus/content/repositories/releases/edu/cmu/tetrad-lib/6.8.0/tetrad-lib-6.8.0.jar -O utils/tetrad-lib-6.8.0.jar
wget https://www.ibiblio.org/xml/XOM/xom-1.3.5.jar -O utils/xom-1.3.5.jar

# compile conversion script
javac utils/convertMag2Pag.java -cp utils/tetrad-lib-6.8.0.jar 
