// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.googlecode.com.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "mesh_headers.hpp"
#include "pncmesh.hpp"

#include <map>
#include <limits>

namespace mfem
{

ParNCMesh::ParNCMesh(MPI_Comm comm, const NCMesh &ncmesh)
   : NCMesh(ncmesh)
{
   MyComm = comm;
   MPI_Comm_size(MyComm, &NRanks);
   MPI_Comm_rank(MyComm, &MyRank);

   // assign leaf elements to the 'NRanks' processors
   for (int i = 0; i < leaf_elements.Size(); i++)
      leaf_elements[i]->rank = InitialPartition(i);

   AssignLeafIndices();
   UpdateVertices();
   PruneGhosts();
}

void ParNCMesh::AssignLeafIndices()
{
   // This is an override of NCMesh::AssignLeafIndices(). The difference is
   // that we don't assign a Mesh index to ghost elements. This will make them
   // skipped in NCMesh::GetMeshComponents.

   for (int i = 0, index = 0; i < leaf_elements.Size(); i++)
   {
      Element* leaf = leaf_elements[i];
      if (leaf->rank == MyRank)
         leaf->index = index++;
      else
         leaf->index = -1;
   }
}

void ParNCMesh::UpdateVertices()
{
   // This is an override of NCMesh::UpdateVertices. This version first
   // assigns Vertex::index to vertices of elements of our rank. Only these
   // vertices then make it to the Mesh in NCMesh::GetMeshComponents.
   // The remaining (ghost) vertices are assigned indices greater or equal to
   // Mesh::GetNV().

   for (HashTable<Node>::Iterator it(nodes); it; ++it)
      if (it->vertex)
         it->vertex->index = -1;

   for (int i = 0; i < leaf_elements.Size(); i++)
   {
      Element* elem = leaf_elements[i];
      if (elem->rank == MyRank)
         for (int j = 0; j < GI[(int) elem->geom].nv; j++)
            elem->node[j]->vertex->index = 0; // mark vertices that we need
   }

   NVertices = 0;
   for (HashTable<Node>::Iterator it(nodes); it; ++it)
      if (it->vertex && it->vertex->index >= 0)
         it->vertex->index = NVertices++;

   vertex_nodeId.SetSize(NVertices);
   for (HashTable<Node>::Iterator it(nodes); it; ++it)
      if (it->vertex && it->vertex->index >= 0)
         vertex_nodeId[it->vertex->index] = it->id;

   NGhostVertices = 0;
   for (HashTable<Node>::Iterator it(nodes); it; ++it)
      if (it->vertex && it->vertex->index < 0)
         it->vertex->index = NVertices + (NGhostVertices++);
}

void ParNCMesh::OnMeshUpdated(Mesh *mesh)
{
   // This is an override (or extension of) NCMesh::OnMeshUpdated().
   // In addition to getting edge/face indices from 'mesh', we also
   // assign indices to ghost edges/faces that don't exist in the 'mesh'.

   // clear Edge:: and Face::index
   for (HashTable<Node>::Iterator it(nodes); it; ++it)
      if (it->edge) it->edge->index = -1;
   for (HashTable<Face>::Iterator it(faces); it; ++it)
      it->index = -1;

   // go assign existing edge/face indices
   NCMesh::OnMeshUpdated(mesh);

   // assign ghost edge indices
   NEdges = mesh->GetNEdges();
   NGhostEdges = 0;
   for (HashTable<Node>::Iterator it(nodes); it; ++it)
      if (it->edge && it->edge->index < 0)
         it->edge->index = NEdges + (NGhostEdges++);

   // assign ghost face indices
   NFaces = mesh->GetNFaces();
   NGhostFaces = 0;
   for (HashTable<Face>::Iterator it(faces); it; ++it)
      if (it->index < 0)
         it->index = NFaces + (NGhostFaces++);
}

void ParNCMesh::ElementSharesEdge(Element *elem, Edge *edge)
{
   // Called by NCMesh::BuildEdgeList when an edge is visited in a leaf element.
   // This allows us to determine edge ownership and processors that share it
   // without duplicating all the HashTable lookups in NCMesh::BuildEdgeList().

   int &owner = edge_owner[edge->index];
   owner = std::min(owner, elem->rank);

   index_rank.Append(Connection(edge->index, elem->rank));
}

void ParNCMesh::ElementSharesFace(Element* elem, Face* face)
{
   // Analogous to ElementHasEdge.

   int &owner = face_owner[face->index];
   owner = std::min(owner, elem->rank);

   index_rank.Append(Connection(face->index, elem->rank));
}

void ParNCMesh::BuildEdgeList()
{
   // This is an extension of NCMesh::BuildEdgeList() which also determines
   // edge ownership, creates edge processor groups and lists shared edges.

   int nedges = NEdges + NGhostEdges;
   edge_owner.SetSize(nedges);
   edge_owner = std::numeric_limits<int>::max();

   index_rank.SetSize(12*leaf_elements.Size() * 3/2);
   index_rank.SetSize(0);

   NCMesh::BuildEdgeList();

   AddSlaveRanks(nedges, edge_list);

   edge_group.MakeFromList(nedges, index_rank);
   index_rank.DeleteAll();

   MakeShared(edge_group, edge_list, shared_edges);
}

void ParNCMesh::BuildFaceList()
{
   // This is an extension of NCMesh::BuildFaceList() which also determines
   // face ownership, creates face processor groups and lists shared faces.

   int nfaces = NFaces + NGhostFaces;
   face_owner.SetSize(nfaces);
   face_owner = std::numeric_limits<int>::max();

   index_rank.SetSize(6*leaf_elements.Size() * 3/2);
   index_rank.SetSize(0);

   NCMesh::BuildFaceList();

   AddSlaveRanks(nfaces, face_list);

   face_group.MakeFromList(nfaces, index_rank);
   index_rank.DeleteAll();

   MakeShared(face_group, face_list, shared_faces);

   CalcFaceOrientations();
}

void ParNCMesh::AddSlaveRanks(int nitems, const NCList& list)
{
   // create a mapping from slave face index to master face index
   Array<int> slave_to_master(nitems);
   slave_to_master = -1;

   for (unsigned i = 0; i < list.slaves.size(); i++)
   {
      const Slave& sf = list.slaves[i];
      slave_to_master[sf.index] = sf.master;
   }

   // We need the groups of master edges/faces to contain the ranks of their
   // slaves (so that master DOFs get sent to those who share the slaves).
   // This can be done by appending more items to 'tmp_ranks' for the masters.
   // (Note that a slave edge can be shared by more than one element/processor.)

   int size = index_rank.Size();
   for (int i = 0; i < size; i++)
   {
      int master = slave_to_master[index_rank[i].from];
      if (master >= 0)
         index_rank.Append(Connection(master, index_rank[i].to));
   }
}

static bool is_shared(const Table& groups, int index, int MyRank)
{
   // An edge/face is shared if its group contains more than one processor and
   // at the same time one of them is ourselves.

   int size = groups.RowSize(index);
   if (size <= 1)
      return false;

   const int* group = groups.GetRow(index);
   for (int i = 0; i < size; i++)
      if (group[i] == MyRank)
         return true;

   return false;
}

void ParNCMesh::MakeShared
(const Table &groups, const NCList &list, NCList &shared)
{
   shared.Clear();

   for (unsigned i = 0; i < list.conforming.size(); i++)
      if (is_shared(groups, list.conforming[i].index, MyRank))
         shared.conforming.push_back(list.conforming[i]);

   for (unsigned i = 0; i < list.masters.size(); i++)
   {
      const Master& master = list.masters[i];
      if (is_shared(groups, master.index, MyRank))
      {
         shared.masters.push_back(master);
         for (int j = master.slaves_begin; j < master.slaves_end; j++)
            shared.slaves.push_back(list.slaves[j]);
      }
   }
}

void ParNCMesh::BuildSharedVertices()
{
   int nvertices = NVertices + NGhostVertices;
   vertex_owner.SetSize(nvertices);
   vertex_owner = std::numeric_limits<int>::max();

   index_rank.SetSize(8*leaf_elements.Size());
   index_rank.SetSize(0);

   Array<MeshId> vertex_id(nvertices);

   // similarly to edges/faces, we loop over the vertices of all leaf elements
   // to determine which processors share each vertex
   for (int i = 0; i < leaf_elements.Size(); i++)
   {
      Element* elem = leaf_elements[i];
      for (int j = 0; j < GI[(int) elem->geom].nv; j++)
      {
         Node* node = elem->node[j];
         int index = node->vertex->index;

         int &owner = vertex_owner[index];
         owner = std::min(owner, elem->rank);

         index_rank.Append(Connection(index, elem->rank));

         MeshId &id = vertex_id[index];
         id.index = (node->edge ? -1 : index);
         id.element = elem;
         id.local = j;
      }
   }

   vertex_group.MakeFromList(nvertices, index_rank);
   index_rank.DeleteAll();

   // create a list of shared vertices, skip obviously slave vertices
   // (for simplicity, we don't guarantee to skip all slave vertices)
   shared_vertices.Clear();
   for (int i = 0; i < nvertices; i++)
   {
      if (is_shared(vertex_group, i, MyRank) && vertex_id[i].index >= 0)
         shared_vertices.conforming.push_back(vertex_id[i]);
   }
}

void ParNCMesh::CalcFaceOrientations()
{
   // Calculate orientation of shared conforming faces.
   // NOTE: face orientation is calculated relative to its lower rank element.

   face_orient.SetSize(NFaces);
   face_orient = 0;

   for (HashTable<Face>::Iterator it(faces); it; ++it)
      if (it->ref_count == 2 && it->index < NFaces)
      {
         Element* e[2] = { it->elem[0], it->elem[1] };
         if (e[0]->rank == e[1]->rank) continue;
         if (e[0]->rank > e[1]->rank) std::swap(e[0], e[1]);

         int ids[2][4];
         for (int i = 0; i < 2; i++)
         {
            int f = find_hex_face(find_node(e[i], it->p1),
                                  find_node(e[i], it->p2),
                                  find_node(e[i], it->p3));

            // get node IDs for the face as seen from e[i]
            const int* fv = GI[Geometry::CUBE].faces[f];
            for (int j = 0; j < 4; j++)
               ids[i][j] = e[i]->node[fv[j]]->id;
         }

         face_orient[it->index] = Mesh::GetQuadOrientation(ids[0], ids[1]);
      }
}

////////////////////////////////////////////////////////////////////////////////

void ParNCMesh::PruneGhosts()
{

}

//// ElementSet ////////////////////////////////////////////////////////////////

void ParNCMesh::ElementSet::SetInt(int pos, int value)
{
   // helper to put an int to the data array
   data[pos] = value & 0xff;
   data[pos+1] = (value >> 8) & 0xff;
   data[pos+2] = (value >> 16) & 0xff;
   data[pos+3] = (value >> 24) & 0xff;
}

int ParNCMesh::ElementSet::GetInt(int pos) const
{
   // helper to get an int from the data array
   return (int) data[pos] +
          ((int) data[pos+1] << 8) +
          ((int) data[pos+2] << 16) +
          ((int) data[pos+3] << 24);
}

bool ParNCMesh::ElementSet::EncodeTree
(Element* elem, const std::set<Element*> &elements)
{
   // is 'elem' in the set?
   if (elements.find(elem) != elements.end())
   {
      // we reached a 'leaf' of our subtree, mark this as zero child mask
      data.Append(0);
      return true;
   }
   else if (elem->ref_type)
   {
      // write a bit mask telling what subtrees contain elements from the set
      int mpos = data.Size();
      data.Append(0);

      // check the subtrees
      int mask = 0;
      for (int i = 0; i < 8; i++)
         if (elem->child[i])
            if (EncodeTree(elem->child[i], elements))
               mask |= (unsigned char) 1 << i;

      if (mask)
         data[mpos] = mask;
      else
         data.DeleteLast();

      return mask != 0;
   }
   return false;
}

ParNCMesh::ElementSet::ElementSet
(const std::set<Element*> &elements, const Array<Element*> &ncmesh_roots)
{
   int header_pos = 0;
   data.SetSize(4);

   // Each refinement tree that contains at least one element from the set
   // is encoded as HEADER + TREE, where HEADER is the root element number and
   // TREE is the output of EncodeTree().
   for (int i = 0; i < ncmesh_roots.Size(); i++)
   {
      if (EncodeTree(ncmesh_roots[i], elements))
      {
         SetInt(header_pos, i);

         // make room for the next header
         header_pos = data.Size();
         data.SetSize(header_pos + 4);
      }
   }

   // mark end of data
   SetInt(header_pos, -1);
}

void ParNCMesh::ElementSet::DecodeTree
(Element* elem, int &pos, Array<Element*> &elements) const
{
   int mask = data[pos++];
   if (!mask)
   {
      elements.Append(elem);
   }
   else
   {
      for (int i = 0; i < 8; i++)
         if (mask & (1 << i))
            DecodeTree(elem->child[i], pos, elements);
   }
}

void ParNCMesh::ElementSet::Decode
(Array<Element*> &elements, const Array<Element*> &ncmesh_roots) const
{
   int root, pos = 0;
   while ((root = GetInt(pos)) >= 0)
   {
      pos += 4;
      DecodeTree(ncmesh_roots[root], pos, elements);
   }
}

template<typename T>
static inline void write(std::ostream& os, T value)
{
   os.write((char*) &value, sizeof(T));
}

template<typename T>
static inline T read(std::istream& is)
{
   T value;
   is.read((char*) &value, sizeof(T));
   return value;
}

void ParNCMesh::ElementSet::Dump(std::ostream &os) const
{
   write<int>(os, data.Size());
   os.write((const char*) data.GetData(), data.Size());
}

void ParNCMesh::ElementSet::Load(std::istream &is)
{
   data.SetSize(read<int>(is));
   is.read((char*) data.GetData(), data.Size());
}


//// EncodeMeshIds/DecodeMeshIds ///////////////////////////////////////////////

void ParNCMesh::EncodeMeshIds(std::ostream &os, Array<MeshId> ids[3]) const
{
   std::map<Element*, int> element_id;

   // get a list of elements involved, dump them to 'os' and create the mapping
   // element_id: (Element* -> stream ID)
   {
      std::set<Element*> elements;
      for (int type = 0; type < 3; type++)
         for (int i = 0; i < ids[type].Size(); i++)
            elements.insert(ids[type][i].element);

      ElementSet eset(elements, root_elements);
      eset.Dump(os);

      Array<Element*> decoded;
      eset.Decode(decoded, root_elements);

      for (int i = 0; i < decoded.Size(); i++)
         element_id[decoded[i]] = i;
   }

   // write the IDs as element/local pairs
   for (int type = 0; type < 3; type++)
   {
      write<int>(os, ids[type].Size());
      for (int i = 0; i < ids[type].Size(); i++)
      {
         const MeshId& id = ids[type][i];
         write<int>(os, element_id[id.element]); // TODO: variable 1-4 bytes
         write<char>(os, id.local);
      }
   }
}

void ParNCMesh::DecodeMeshIds(std::istream &is, Array<MeshId> ids[3]) const
{
   // read the list of elements
   ElementSet eset(is);

   Array<Element*> elements;
   eset.Decode(elements, root_elements);

   // read vertex/edge/face IDs
   for (int type = 0; type < 3; type++)
   {
      int ne = read<int>(is);
      ids[type].SetSize(ne);

      for (int i = 0; i < ne; i++)
      {
         Element* elem = elements[read<int>(is)];
         MFEM_ASSERT(!elem->ref_type, "Not a leaf element.");

         MeshId &id = ids[type][i];
         id.element = elem;
         id.local = read<char>(is);

         // find vertex/edge/face index
         GeomInfo &gi = GI[(int) elem->geom];
         switch (type)
         {
            case 0:
            {
               id.index = elem->node[id.local]->vertex->index;
               break;
            }
            case 1:
            {
               const int* ev = gi.edges[id.local];
               Node* node = nodes.Peek(elem->node[ev[0]], elem->node[ev[1]]);
               MFEM_ASSERT(node && node->edge, "Edge not found.");
               id.index = node->edge->index;
               break;
            }
            default:
            {
               const int* fv = gi.faces[id.local];
               Face* face = faces.Peek(elem->node[fv[0]], elem->node[fv[1]],
                                       elem->node[fv[2]], elem->node[fv[3]]);
               MFEM_ASSERT(face, "Face not found.");
               id.index = face->index;
            }
         }
      }
   }
}

//// Messages //////////////////////////////////////////////////////////////////

void NeighborDofMessage::AddDofs
(int type, const NCMesh::MeshId &id, const Array<int> &dofs, ParNCMesh* pncmesh)
{
   MFEM_ASSERT(type >= 0 && type < 3, "");
   id_dofs[type][id].assign(dofs.GetData(), dofs.GetData() + dofs.Size());
   this->pncmesh = pncmesh;
}

void NeighborDofMessage::GetDofs
(int type, const NCMesh::MeshId& id, Array<int>& dofs)
{
   MFEM_ASSERT(type >= 0 && type < 3, "");
#ifdef MFEM_DEBUG
   if (id_dofs[type].find(id) == id_dofs[type].end())
      MFEM_ABORT("Type/ID " << type << "/" << id.index
                 << " not found in neighbor message.");
#endif
   std::vector<int> &vec = id_dofs[type][id];
   dofs.SetSize(vec.size());
   dofs.Assign(vec.data());
}

void NeighborDofMessage::ReorderEdgeDofs
(const NCMesh::MeshId &id, std::vector<int> &dofs)
{
   // Reorder the DOFs into/from a neutral ordering, independent of local
   // edge orientation.

   const int *ev = NCMesh::GI[(int) id.element->geom].edges[id.local];
   int v0 = id.element->node[ev[0]]->vertex->index;
   int v1 = id.element->node[ev[1]]->vertex->index;

   if ((v0 < v1 && ev[0] > ev[1]) || (v0 > v1 && ev[0] < ev[1]))
   {
      // "invert" the edge DOFs
      // FIXME: assuming nv == 1
      std::swap(dofs[0], dofs[1]);
      int n = dofs.size() - 2;
      for (int i = 0; i < n/2; i++)
         std::swap(dofs[i + 2], dofs[dofs.size()-1 - i]);
   }
}

static void write_dofs(std::ostream &os, const std::vector<int> &dofs)
{
   write<int>(os, dofs.size());
   // TODO: we should compress the ints, mostly they are contiguous ranges
   os.write((const char*) dofs.data(), dofs.size() * sizeof(int));
}

static void read_dofs(std::istream &is, std::vector<int> &dofs)
{
   dofs.resize(read<int>(is));
   is.read((char*) dofs.data(), dofs.size() * sizeof(int));
}

void NeighborDofMessage::Encode()
{
   IdToDofs::iterator it;

   // collect vertex/edge/face IDs
   Array<NCMesh::MeshId> ids[3];
   for (int type = 0; type < 3; type++)
   {
      ids[type].Reserve(id_dofs[type].size());
      for (it = id_dofs[type].begin(); it != id_dofs[type].end(); ++it)
         ids[type].Append(it->first);
   }

   // encode the IDs
   std::ostringstream stream;
   pncmesh->EncodeMeshIds(stream, ids);

   // dump the DOFs
   for (int type = 0; type < 3; type++)
   {
      for (it = id_dofs[type].begin(); it != id_dofs[type].end(); ++it)
      {
         if (type == 1) ReorderEdgeDofs(it->first, it->second);
         write_dofs(stream, it->second);
      }

      // no longer need the original data
      id_dofs[type].clear();
   }

   stream.str().swap(data);
}

void NeighborDofMessage::Decode()
{
   std::istringstream stream(data);

   // decode vertex/edge/face IDs
   Array<NCMesh::MeshId> ids[3];
   pncmesh->DecodeMeshIds(stream, ids);

   // load DOFs
   for (int type = 0; type < 3; type++)
   {
      id_dofs[type].clear();
      for (int i = 0; i < ids[type].Size(); i++)
      {
         const NCMesh::MeshId &id = ids[type][i];
         read_dofs(stream, id_dofs[type][id]);
         if (type == 1) ReorderEdgeDofs(id, id_dofs[type][id]);
      }
   }

   // no longer need the raw data
   data.clear();
}

void NeighborRowRequest::Encode()
{
   std::ostringstream stream;

   // write the int set to the stream
   write<int>(stream, rows.size());
   for (std::set<int>::iterator it = rows.begin(); it != rows.end(); ++it)
      write<int>(stream, *it);

   rows.clear();
   stream.str().swap(data);
}

void NeighborRowRequest::Decode()
{
   std::istringstream stream(data);

   // read the int set from the stream
   rows.clear();
   int size = read<int>(stream);
   for (int i = 0; i < size; i++)
      rows.insert(rows.end(), read<int>(stream));

   data.clear();
}

void NeighborRowReply::AddRow
(int row, const Array<int> &cols, const Vector &srow)
{
   Row& row_data = rows[row];
   row_data.cols.assign(cols.GetData(), cols.GetData() + cols.Size());
   row_data.srow = srow;
}

void NeighborRowReply::GetRow(int row, Array<int> &cols, Vector &srow)
{
#ifdef MFEM_DEBUG
   if (rows.find(row) == rows.end())
      MFEM_ABORT("Row " << row << " not found in neighbor message.");
#endif
   Row& row_data = rows[row];
   cols.SetSize(row_data.cols.size());
   cols.Assign(row_data.cols.data());
   srow = row_data.srow;
}

void NeighborRowReply::Encode()
{
   std::ostringstream stream;

   // dump the rows to the stream
   write<int>(stream, rows.size());
   for (std::map<int, Row>::iterator it = rows.begin(); it != rows.end(); ++it)
   {
      write<int>(stream, it->first); // row number
      Row& row_data = it->second;
      MFEM_ASSERT((int) row_data.cols.size() == row_data.srow.Size(), "");
      write_dofs(stream, row_data.cols);
      stream.write((const char*) row_data.srow.GetData(),
                   sizeof(double) * row_data.srow.Size());
   }

   rows.clear();
   stream.str().swap(data);
}

void NeighborRowReply::Decode()
{
   std::istringstream stream(data);

   // NOTE: there is no rows.clear() since a row reply can be received
   // repeatedly and the received rows accumulate.

   // read the rows
   int size = read<int>(stream);
   for (int i = 0; i < size; i++)
   {
      Row& row_data = rows[read<int>(stream)];
      read_dofs(stream, row_data.cols);
      row_data.srow.SetSize(row_data.cols.size());
      stream.read((char*) row_data.srow.GetData(),
                  sizeof(double) * row_data.srow.Size());

      /*std::cout << "Received row: ";
      for (int j = 0; j < row_data.cols.size(); j++)
         std::cout << "(" << row_data.cols[j] << "," << row_data.srow(j) << ")";
      std::cout << std::endl;*/
   }

   data.clear();
}

} // namespace mfem

#endif // MFEM_USE_MPI
