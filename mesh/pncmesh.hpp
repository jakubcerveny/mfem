#ifndef MFEM_PNCMESH
#define MFEM_PNCMESH

#include "../config.hpp"

#ifdef MFEM_USE_MPI

#include <map>
#include <set>

#include "ncmesh.hpp"
#include "../general/communication.hpp"

namespace mfem
{

/** \brief
 *
 */
class ParNCMesh : public NCMesh
{
public:
   ParNCMesh(MPI_Comm comm, const NCMesh& ncmesh);

   /** Return a list of vertices shared by this processor and at least one other
       processor. (NOTE: only NCList::conforming will be set.) */
   const NCList& GetSharedVertices()
   {
      if (shared_vertices.Empty()) BuildSharedVertices();
      return shared_vertices;
   }

   /** Return a list of edges shared by this processor and at least one other
       processor. (NOTE: this is a subset of the NCMesh::edge_list; slaves are
       empty.) */
   const NCList& GetSharedEdges()
   {
      if (edge_list.Empty()) BuildEdgeList();
      return shared_edges;
   }

   /** Return a list of faces shared by this processor and another processor.
       (NOTE: this is a subset of NCMesh::face_list; slaves are empty.) */
   const NCList& GetSharedFaces()
   {
      if (face_list.Empty()) BuildFaceList();
      return shared_faces;
   }

   /// Helper to get shared vertices/edges/faces ('type' == 0/1/2 resp.).
   const NCList& GetSharedList(int type)
   {
      switch (type)
      {
      case 0: return GetSharedVertices();
      case 1: return GetSharedEdges();
      default: return GetSharedFaces();
      }
   }

   /// Return (shared) edge/face ('type' == 1/2) orientation.
   int GetOrientation(int type, int index)
   {
      switch (type)
      {
      case 0: return 0;
      case 1: return edge_orient[index];
      default: return face_orient[index];
      }
   }

   /// Return vertex/edge/face ('type' == 0/1/2, resp.) owner.
   int GetOwner(int type, int index)
   {
      switch (type)
      {
      case 0: return vertex_owner[index];
      case 1: return edge_owner[index];
      default: return face_owner[index];
      }
   }

   /** Return a list of processors sharing a vertex/edge/face
       ('type' == 0/1/2, resp.) and the size of the list. */
   const int* GetGroup(int type, int index, int &size) const
   {
      switch (type)
      {
      case 0: size = vertex_group.RowSize(index); return vertex_group.GetRow(index);
      case 1: size = edge_group.RowSize(index); return edge_group.GetRow(index);
      default: size = face_group.RowSize(index); return face_group.GetRow(index);
      }
   }

   /** */
   bool RankInGroup(int type, int index, int rank) const
   {
      int size;
      const int* group = GetGroup(type, index, size);
      for (int i = 0; i < size; i++)
         if (group[i] == rank)
            return true;
      return false;
   }

   /** */
   bool IsGhost(int type, int index) const
   {
      switch (type)
      {
      case 0: return index >= NVertices;
      case 1: return index >= NEdges;
      case 2: return index >= NFaces;
      }
   }

protected:
   MPI_Comm MyComm;
   int NRanks, MyRank;

   int NVertices, NGhostVertices;
   int NEdges, NGhostEdges;
   int NFaces, NGhostFaces;

   //
   NCList shared_vertices;
   NCList shared_edges;
   NCList shared_faces;

   //
   Array<int> vertex_owner;
   Array<int> edge_owner;
   Array<int> face_owner;

   //
   Table vertex_group;
   Table edge_group;
   Table face_group;

   Array<char> edge_orient;
   Array<char> face_orient;

   void InitialPartition();

   virtual void UpdateVertices();
   virtual void AssignLeafIndices();

   virtual void OnMeshUpdated(Mesh *mesh);

   virtual void BuildEdgeList();
   virtual void BuildFaceList();

   virtual void ElementSharesEdge(Element* elem, Edge* edge);
   virtual void ElementSharesFace(Element* elem, Face* face);

   void BuildSharedVertices();

   void CalcEdgeOrientations();
   void CalcFaceOrientations();

   /// Struct to help sorting edges/faces
   struct IndexRank
   {
      int index, rank;
      IndexRank(int index, int rank) : index(index), rank(rank) {}
      bool operator< (const IndexRank &rhs) const
      { return (index == rhs.index) ? (rank < rhs.rank) : (index < rhs.index); }
      bool operator== (const IndexRank &rhs) const
      { return (index == rhs.index) && (rank == rhs.rank); }
   };

   Array<IndexRank> tmp_ranks;

   void AddSlaveRanks(int nitems, const NCList& list);
   void MakeGroups(int nitems, Table &groups);
   void MakeShared(const Table &groups, const NCList &list, NCList &shared);

   /** Uniquely encodes a set of elements in the erefinement hierarchy of an
       NCMesh. Can be dumped to a stream, sent to another processor, loaded,
       and decoded to identify the same set of elements (refinements) in a
       different but compatible NCMesh. The elements don't have to be leaves,
       but they must represent subtrees of 'ncmesh_roots'. */
   class ElementSet
   {
   public:
      ElementSet(const std::set<Element*> &elements,
                 const Array<Element*> &ncmesh_roots);
      void Dump(std::ostream &os) const;

      ElementSet() {}
      ElementSet(std::istream &is) { Load(is); }
      void Load(std::istream &is);
      void Decode(Array<Element*> &elements,
                  const Array<Element*> &ncmesh_roots) const;

   protected:
      Array<unsigned char> data; ///< encoded refinement (sub-)trees

      bool EncodeTree(Element* elem, const std::set<Element*> &elements);
      void DecodeTree(Element* elem, int &pos, Array<Element*> &elements) const;

      void SetInt(int pos, int value);
      int GetInt(int pos) const;
   };

   /// Write to 'os' a processor-independent encoding of vertex/edge/face IDs.
   void EncodeMeshIds(std::ostream &os, Array<MeshId> ids[3]) const;

   /// Read from 'is' a processor-independent encoding of vetex/edge/face IDs.
   void DecodeMeshIds(std::istream &is, Array<MeshId> ids[3]) const;

   friend class ParMesh;
   friend class NeighborDofMessage;
};

/*
TODO
+ shared vertices
+ fix NCMesh::BuildEdgeList
+ AddSlaveDependencies, AddOneOneDependencies
+ Phase 2
- invalid CDOFs?
+ hypre matrix
+ essential BC
+ mask slave DOFs
+ MakeRef bug
- shared face orientation
*/


/// Variable-length MPI message containing unspecific binary data.
template<int Tag>
struct VarMessage
{
   std::string data;
   MPI_Request send_request;

   /// Non-blocking send to processor 'rank'.
   void Isend(int rank, MPI_Comm comm)
   {
      Encode();
      MPI_Isend(data.data(), data.length(), MPI_BYTE, rank, Tag, comm,
                &send_request);
   }

   /// Helper to send all messages in a rank-to-message map container.
   template<typename MapT>
   static void IsendAll(MapT& rank_msg, MPI_Comm comm) // TODO: free function instead of static?
   {
      typename MapT::iterator it;
      for (it = rank_msg.begin(); it != rank_msg.end(); ++it)
         it->second.Isend(it->first, comm);
   }

   /// Helper to wait for all messages in a map container to be sent.
   template<typename MapT>
   static void WaitAllSent(MapT& rank_msg) // TODO: free function instead of static?
   {
      typename MapT::iterator it;
      for (it = rank_msg.begin(); it != rank_msg.end(); ++it)
      {
         MPI_Wait(&it->second.send_request, MPI_STATUS_IGNORE);
         it->second.Clear();
      }
   }

   /** Blocking probe for incoming message of this type from any rank.
       Returns the rank and message size. */
   static void Probe(int &rank, int &size, MPI_Comm comm)
   {
      MPI_Status status;
      MPI_Probe(MPI_ANY_SOURCE, Tag, comm, &status);
      rank = status.MPI_SOURCE;
      MPI_Get_count(&status, MPI_BYTE, &size);
   }

   /** Non-blocking probe for incoming message of this type from any rank.
       If there is an incoming message, returns true and sets 'rank' and 'size'.
       Otherwise returns false. */
   static bool IProbe(int &rank, int &size, MPI_Comm comm)
   {
      int flag;
      MPI_Status status;
      MPI_Iprobe(MPI_ANY_SOURCE, Tag, comm, &flag, &status);
      if (!flag) return false;

      rank = status.MPI_SOURCE;
      MPI_Get_count(&status, MPI_BYTE, &size);
      return true;
   }

   /// Post-probe receive from processor 'rank' of message size 'size'.
   void Recv(int rank, int size, MPI_Comm comm)
   {
      data.resize(size);
      MPI_Status status;
      MPI_Recv((void*) data.data(), size, MPI_BYTE, rank, Tag, comm, &status);
      Decode();
   }

   /// Helper to receive all messages in a rank-to-message map container.
   template<typename MapT>
   static void RecvAll(MapT& rank_msg, MPI_Comm comm) // TODO: free function instead of static?
   {
      int recv_left = rank_msg.size();
      while (recv_left > 0)
      {
         int rank, size;
         Probe(rank, size, comm);
         rank_msg[rank].Recv(rank, size, comm);
         --recv_left;
      }
   }

   VarMessage() : send_request(MPI_REQUEST_NULL) {}
   void Clear() { data.clear(); send_request = MPI_REQUEST_NULL; }

protected:
   virtual void Encode() {}
   virtual void Decode() {}
};


/** */
class NeighborDofMessage : public VarMessage<135>
{
public:
   /// Add vertex/edge/face DOFs to an outgoing message.
   void AddDofs(int type, const NCMesh::MeshId &id, const Array<int> &dofs,
                ParNCMesh* pncmesh);

   /// Set pointer to ParNCMesh (needed to encode the message).
   void SetNCMesh(ParNCMesh* pncmesh) { this->pncmesh = pncmesh; }

   /// Get vertex/edge/face DOFs from a received message.
   void GetDofs(int type, const NCMesh::MeshId& id, Array<int>& dofs);

   typedef std::map<int, NeighborDofMessage> Map;

protected:
   // TODO: we need a Dictionary class if we want to avoid STL
   typedef std::map<NCMesh::MeshId, std::vector<int> > IdToDofs;
   IdToDofs id_dofs[3];

   ParNCMesh* pncmesh;
   virtual void Encode();
   virtual void Decode();
};


/** */
class NeighborRowRequest: public VarMessage<312>
{
public:
   std::set<int> rows;

   void RequestRow(int row) { rows.insert(row); }
   void RemoveRequest(int row) { rows.erase(row); }

   typedef std::map<int, NeighborRowRequest> Map;

protected:
   virtual void Encode();
   virtual void Decode();
};


/** */
class NeighborRowReply: public VarMessage<313>
{
public:
   void AddRow(int row, const Array<int> &cols, const Vector &srow);

   bool HaveRow(int row) const { return rows.find(row) != rows.end(); }
   void GetRow(int row, Array<int> &cols, Vector &srow);

   typedef std::map<int, NeighborRowReply> Map;

protected:
   struct Row { std::vector<int> cols; Vector srow; };
   std::map<int, Row> rows;

   virtual void Encode();
   virtual void Decode();
};


// comparison operator so that MeshId can be used as key in std::map
inline bool operator< (const NCMesh::MeshId &a, const NCMesh::MeshId &b)
{ return a.index < b.index; }


} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_PNCMESH
