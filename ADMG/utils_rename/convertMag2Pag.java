// package utils;

import edu.cmu.tetrad.data.*;
import edu.cmu.tetrad.graph.*;
import edu.cmu.tetrad.search.*;
import java.io.File;

public class convertMag2Pag {

    private static void convert(String path){
        File graph = new File(path);

        if (!graph.getName().endsWith(".mag")){
            System.out.println(graph.getName().concat(" --- DOES NOT EXIST"));
            return;
        }

        try {
            Graph mag = GraphUtils.loadGraphTxt(graph);
            MagToPag obj = new MagToPag(mag);
            obj.setCompleteRuleSetUsed(true);
            Graph pag = obj.convert();
            File out = new File(graph.getParent().concat("/").concat(graph.getName()).concat(".pag"));
            GraphUtils.saveGraph(pag, out, false);
        }

        catch(Exception e){
            System.out.println(graph.getName().concat(" --- ERROR"));
        }
    }

    public static void main(String[] args) {
        convert(args[0]);
    }
}