/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * MarkdownTextPane.java
 * Copyright (C) 2015 University of Waikato, Hamilton, NZ
 */

package meka.gui.core;

import com.googlecode.jfilechooserbookmarks.gui.BasePanel;
import com.googlecode.jfilechooserbookmarks.gui.BaseScrollPane;
import com.petebevin.markdown.MarkdownProcessor;

import java.awt.BorderLayout;

import javax.swing.JEditorPane;
import javax.swing.JFrame;
import javax.swing.text.Document;

/**
 * Renders Markdown text.
 *
 * @author FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision$
 */
public class MarkdownTextPane extends BasePanel {

  private static final long serialVersionUID = -3021897813785552183L;

  /** the markdown processor. */
  protected MarkdownProcessor m_Processor;

  /** the markdown text. */
  protected String m_Markdown;

  /** for rendering the markdown. */
  protected JEditorPane m_PaneView;

  /**
   * Initializes the members.
   */
  @Override
  protected void initialize() {
    super.initialize();

    this.m_Processor = new MarkdownProcessor();
    this.m_Markdown = "";
  }

  /**
   * Initializes the widgets.
   */
  @Override
  protected void initGUI() {
    super.initGUI();

    this.setLayout(new BorderLayout());

    this.m_PaneView = new JEditorPane();
    this.m_PaneView.setEditable(false);
    this.m_PaneView.setContentType("text/html");
    this.add(new BaseScrollPane(this.m_PaneView), BorderLayout.CENTER);
  }

  /**
   * Sets the Markdown text and renders it.
   *
   * @param value
   *          the markdown text
   */
  public void setText(String value) {
    String html;

    if (value == null) {
      value = "";
    }
    this.m_Markdown = value;

    html = this.m_Processor.markdown(this.m_Markdown);
    try {
      this.m_PaneView.setText("<html>" + html + "</html>");
      this.m_PaneView.setCaretPosition(0);
    } catch (Exception e) {
      System.err.println("Failed to update preview!");
      e.printStackTrace();
    }
  }

  /**
   * Returns the Markdown text.
   *
   * @return the markdown text
   */
  public String getText() {
    return this.m_Markdown;
  }

  /**
   * Sets whether the text pane is editable or not.
   *
   * @param value
   *          if true the text pane is editable
   */
  public void setEditable(final boolean value) {
    this.m_PaneView.setEditable(value);
  }

  /**
   * Returns whether the text pane is editable or not.
   *
   * @return true if the text pane is editable
   */
  public boolean isEditable() {
    return this.m_PaneView.isEditable();
  }

  /**
   * Returns the underlying document.
   *
   * @return the document
   */
  public Document getDocument() {
    return this.m_PaneView.getDocument();
  }

  /**
   * Sets the position of the cursor.
   *
   * @param value
   *          the position
   */
  public void setCaretPosition(final int value) {
    this.m_PaneView.setCaretPosition(value);
  }

  /**
   * Returns the current position of the cursor.
   *
   * @return the cursor position
   */
  public int getCaretPosition() {
    return this.m_PaneView.getCaretPosition();
  }

  /**
   * Sets the position of the cursor at the end.
   */
  public void setCaretPositionLast() {
    this.setCaretPosition(this.getDocument().getLength());
  }

  /**
   * For testing only.
   *
   * @param args
   *          ignored
   */
  public static void main(final String[] args) {
    MarkdownTextPane pane = new MarkdownTextPane();
    pane.setText("# Markdown test\n\n* item 1\n* item 2\n\n## Other stuff\n*italic* __bold__");
    JFrame frame = new JFrame("Markdown test");
    frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    frame.getContentPane().setLayout(new BorderLayout());
    frame.getContentPane().add(pane, BorderLayout.CENTER);
    frame.setSize(600, 400);
    frame.setLocationRelativeTo(null);
    frame.setVisible(true);
  }
}
