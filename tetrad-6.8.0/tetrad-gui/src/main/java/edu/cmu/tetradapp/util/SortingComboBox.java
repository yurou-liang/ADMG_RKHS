///////////////////////////////////////////////////////////////////////////////
// For information as to what this class does, see the Javadoc, below.       //
// Copyright (C) 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006,       //
// 2007, 2008, 2009, 2010, 2014, 2015 by Peter Spirtes, Richard Scheines, Joseph   //
// Ramsey, and Clark Glymour.                                                //
//                                                                           //
// This program is free software; you can redistribute it and/or modify      //
// it under the terms of the GNU General Public License as published by      //
// the Free Software Foundation; either version 2 of the License, or         //
// (at your option) any later version.                                       //
//                                                                           //
// This program is distributed in the hope that it will be useful,           //
// but WITHOUT ANY WARRANTY; without even the implied warranty of            //
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             //
// GNU General Public License for more details.                              //
//                                                                           //
// You should have received a copy of the GNU General Public License         //
// along with this program; if not, write to the Free Software               //
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA //
///////////////////////////////////////////////////////////////////////////////

package edu.cmu.tetradapp.util;

import javax.swing.*;

/**
 * Extends JComboBox so that the items it contains are automatically sorted as
 * they are added.
 *
 * @author Joseph Ramsey
 */
public class SortingComboBox extends JComboBox {

    /**
     * Overrides the addItem() method of JComboBox to automatically sort items
     * as they are added.  Items are sorted by the strings they return in their
     * toString() methods.
     *
     * @param anItem the item to add.
     */
    public void addItem(Object anItem) {

        String name = anItem.toString();

        for (int i = 0; i < getItemCount(); i++) {
            Object o = getItemAt(i);
            String oName = o.toString();

            if (oName.compareTo(name) > 0) {
                insertItemAt(anItem, i);

                return;
            }
        }

        super.addItem(anItem);
    }
}





