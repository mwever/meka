package meka;

import java.io.File;
import java.io.FileReader;

import meka.classifiers.multilabel.BCC;
import meka.classifiers.multilabel.BPNN;
import meka.classifiers.multilabel.BR;
import meka.classifiers.multilabel.BRq;
import meka.classifiers.multilabel.CC;
import meka.classifiers.multilabel.CCq;
import meka.classifiers.multilabel.CDN;
import meka.classifiers.multilabel.CDT;
import meka.classifiers.multilabel.CT;
import meka.classifiers.multilabel.DBPNN;
import meka.classifiers.multilabel.FW;
import meka.classifiers.multilabel.HASEL;
import meka.classifiers.multilabel.LC;
import meka.classifiers.multilabel.MCC;
import meka.classifiers.multilabel.MLCBMaD;
import meka.classifiers.multilabel.MajorityLabelset;
import meka.classifiers.multilabel.MultiLabelClassifier;
import meka.classifiers.multilabel.PCC;
import meka.classifiers.multilabel.PLST;
import meka.classifiers.multilabel.PMCC;
import meka.classifiers.multilabel.PS;
import meka.classifiers.multilabel.PSt;
import meka.classifiers.multilabel.RAkEL;
import meka.classifiers.multilabel.RAkELd;
import meka.classifiers.multilabel.RT;
import meka.classifiers.multilabel.meta.BaggingML;
import meka.classifiers.multilabel.meta.BaggingMLdup;
import meka.classifiers.multilabel.meta.CM;
import meka.classifiers.multilabel.meta.DeepML;
import meka.classifiers.multilabel.meta.EM;
import meka.classifiers.multilabel.meta.EnsembleML;
import meka.classifiers.multilabel.meta.MBR;
import meka.classifiers.multilabel.meta.MultiSearch;
import meka.classifiers.multilabel.meta.RandomSubspaceML;
import meka.classifiers.multilabel.meta.SubsetMapper;
import meka.core.MLUtils;
import weka.core.Instances;

public class TestTimeouts {

  public static void main(final String[] args) throws Exception {

    MultiLabelClassifier[] mlcs = { new BaggingML(), new BaggingMLdup(), new CM(), new DeepML(), new EM(), new EnsembleML(), new MBR(), new MultiSearch(), new RandomSubspaceML(),
        new SubsetMapper(), new BCC(), new BPNN(), new BR(), new BRq(), new CC(), new CCq(), new CDN(), new CDT(), new CT(), new DBPNN(), new FW(), new HASEL(), new LC(),
        new MajorityLabelset(), new MCC(), new MLCBMaD(), new PCC(), new PLST(), new PMCC(), new PS(), new PSt(), new RAkEL(), new RAkELd(), new RT() };

    Instances data = new Instances(new FileReader(new File("../datasets/classification/multi-label/enron-f.arff")));
    MLUtils.prepareData(data);

    for (MultiLabelClassifier mlc : mlcs) {
      System.out.println("Evaluating timeout of " + mlc.getClass().getSimpleName());
      Thread t = new Thread() {
        @Override
        public void run() {
          try {
            System.out.println(Thread.currentThread().getName() + " Start training " + mlc.getClass().getSimpleName());
            mlc.buildClassifier(data);
          } catch (InterruptedException e) {
            System.out.println(Thread.currentThread().getName() + "Interrupt worked :)");
          } catch (Exception e) {
            e.printStackTrace();
          }
        }
      };

      t.start();
      Thread.sleep(1 * 1000);
      System.out.println("Interrupt training " + mlc.getClass().getSimpleName());
      t.interrupt();
      t.join();
      System.out.println();
    }

  }

}
